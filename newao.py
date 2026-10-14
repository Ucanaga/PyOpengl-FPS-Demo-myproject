"""Tek dosyalik FPS test ortami: OpenGL 3.3, shadow mapping, SSAO ve probe GI."""
import math, sys, time, ctypes
import numpy as np
import glfw
from OpenGL.GL import *
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
W, H = (1280, 720)
PLAYER = np.array([0.0, 0.0, 14.0], dtype=np.float32)
PLAYER_RADIUS, PLAYER_HEIGHT, EYE = (0.38, 1.75, 1.6)
YAW, PITCH, SENS, SPEED = (-90.0, 0.0, 0.12, 6.0)
LOCKED, FIRST_MOUSE, LAST_X, LAST_Y, Q_PREV = (True, True, W / 2, H / 2, False)
LIGHT_COLOR = [1.0, 0.96, 0.86]
LIGHT_POWER, LIGHT_AZ, LIGHT_EL = (1.4, 45.0, 55.0)
AMBIENT = [0.12, 0.14, 0.17]
SHADOWS, AO_ON, GI_ON = (True, True, True)
AO_RADIUS, AO_BIAS, AO_STRENGTH = (0.65, 0.025, 0.8)
GI_STRENGTH, GI_BOUNCES = (0.65, 1)

# -----------------------------------------------------------------------------
# Math and matrix helpers
# -----------------------------------------------------------------------------
def norm(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-08 else v

def look_at(eye, target, up):
    eye, target, up = map(lambda x: np.asarray(x, np.float32), (eye, target, up))
    f = norm(target - eye)
    s = norm(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = (s, u, -f)
    m[0, 3], m[1, 3], m[2, 3] = (-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye))
    return m

def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = (f / aspect, f)
    m[2, 2], m[2, 3], m[3, 2] = ((far + near) / (near - far), 2 * far * near / (near - far), -1)
    return m

def ortho(l, r, b, t, n, f):
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = (2 / (r - l), 2 / (t - b), -2 / (f - n))
    m[0, 3], m[1, 3], m[2, 3] = (-(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n))
    return m

def model_mat(pos, half):
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = (2 * half[0], 2 * half[1], 2 * half[2])
    m[:3, 3] = pos
    return m

def set_mat(p, name, m):
    glUniformMatrix4fv(glGetUniformLocation(p, name), 1, GL_TRUE, np.asarray(m, np.float32))

def set1i(p, n, v):
    glUniform1i(glGetUniformLocation(p, n), int(v))

def set1f(p, n, v):
    glUniform1f(glGetUniformLocation(p, n), float(v))

def set3(p, n, v):
    glUniform3f(glGetUniformLocation(p, n), *map(float, v))

def shader(src, typ):
    s = glCreateShader(typ)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode(errors='replace'))
    return s

def program(vs, fs):
    a, b = (shader(vs, GL_VERTEX_SHADER), shader(fs, GL_FRAGMENT_SHADER))
    p = glCreateProgram()
    glAttachShader(p, a)
    glAttachShader(p, b)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode(errors='replace'))
    glDeleteShader(a)
    glDeleteShader(b)
    return p
# -----------------------------------------------------------------------------
# OpenGL shaders
# -----------------------------------------------------------------------------
MAIN_VS = '''
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 model,view,projection,lightSpace;
out vec3 worldPos;
out vec3 worldNormal;
out vec4 lightPos;
void main() {
    vec4 w=model*vec4(aPos,1);
    worldPos=w.xyz;
    worldNormal=mat3(transpose(inverse(model)))*aNormal;
    lightPos=lightSpace*w;
    gl_Position=projection*view*w;
}
'''
MAIN_FS = '''
#version 330 core
in vec3 worldPos;
in vec3 worldNormal;
in vec4 lightPos;
layout(location=0) out vec4 colorOut;
layout(location=1) out vec4 normalOut;
uniform vec3 lightDir,lightColor,ambientColor,objectColor;
uniform mat4 view;
uniform sampler2D shadowMap;
uniform bool shadowsEnabled,giEnabled;
uniform float shadowBias,giStrength;
uniform sampler3D giTex;
uniform vec3 giMin,giMax;
uniform ivec3 giGrid;
float shadowing(vec3 N,vec3 L) {
    vec3 q=lightPos.xyz/lightPos.w*0.5+0.5;
    if(q.z>1.0||q.x<0.0||q.x>1.0||q.y<0.0||q.y>1.0)return 0.0;
    float b=max(shadowBias*(1.0-dot(N,L)),shadowBias*0.35);
    vec2 ts=1.0/vec2(textureSize(shadowMap,0));
    float s=0.0;
    for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)s+=(q.z-b>texture(shadowMap,q.xy+vec2(x,y)*ts).r)?1.0:0.0;
    return s/9.0;
}
void main() {
    vec3 N=normalize(worldNormal), L=normalize(-lightDir);
    float nd=max(dot(N,L),0.0);
    float sh=shadowsEnabled?shadowing(N,L):0.0;
    vec3 base=ambientColor*objectColor+(1.0-sh)*nd*lightColor*objectColor;
    if(giEnabled) {
        vec3 uv=clamp((worldPos-giMin)/(giMax-giMin),0.0,1.0);
        vec3 bounce=texture(giTex,uv).rgb;
        base+=bounce*objectColor*giStrength;
    }
    colorOut=vec4(base,1);
    normalOut=vec4(normalize(mat3(view)*N),1);
}
'''
DEPTH_VS = '''
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 model,lightSpace;
void main() {
    gl_Position=lightSpace*model*vec4(aPos,1);
}
'''
DEPTH_FS = '''
#version 330 core
void main() {
}
'''
QUAD_VS = '''
#version 330 core
layout(location=0) in vec2 p;
layout(location=1) in vec2 uv;
out vec2 vUV;
void main() {
    vUV=uv;
    gl_Position=vec4(p,0,1);
}
'''
SSAO_FS = '''
#version 330 core
in vec2 vUV;
out float aoOut;
uniform sampler2D gDepth,gNormal,noiseTex;
uniform mat4 projection,inverseProjection;
uniform vec3 samples[16];
uniform float radius,bias;
uniform vec2 noiseScale;
vec3 posAt(vec2 uv,float d) {
    vec4 p=inverseProjection*vec4(uv*2-1,d*2-1,1);
    return p.xyz/p.w;
}
void main() {
    float d=texture(gDepth,vUV).r;
    if(d>=0.99999) {
        aoOut=1;
        return;
    }
    vec3 n=normalize(texture(gNormal,vUV).xyz);
    vec3 p=posAt(vUV,d);
    vec3 rnd=texture(noiseTex,vUV*noiseScale).xyz*2-1;
    vec3 t=rnd-n*dot(rnd,n);
    if(dot(t,t)<1e-5)t=vec3(1,0,0);
    t=normalize(t);
    mat3 T=mat3(t,normalize(cross(n,t)),n);
    float occ=0;
    for(int i=0;i<16;i++) {
        vec3 q=p+T*samples[i]*radius;
        vec4 clip=projection*vec4(q,1);
        clip.xyz/=clip.w;
        vec2 uv=clip.xy*0.5+0.5;
        if(any(lessThan(uv,vec2(0)))||any(greaterThan(uv,vec2(1))))continue;
        float sd=texture(gDepth,uv).r;
        if(sd>=0.99999)continue;
        vec3 sp=posAt(uv,sd);
        float range=smoothstep(0,1,radius/max(abs(p.z-sp.z),0.0001));
        occ+=(sp.z>=q.z+bias?1.0:0.0)*range;
    }
    aoOut=clamp(1-occ/16.0,0,1);
}
'''
BLUR_FS = '''
#version 330 core
in vec2 vUV;
out float o;
uniform sampler2D src;
uniform vec2 texel;
void main() {
    float s=0;
    float w=0;
    for(int x=-2;x<=2;x++)for(int y=-2;y<=2;y++) {
        float k=1.0/(1.0+abs(x)+abs(y));
        s+=texture(src,vUV+vec2(x,y)*texel).r*k;
        w+=k;
    }
    o=s/w;
}
'''
COMP_FS = '''
#version 330 core
in vec2 vUV;
out vec4 outColor;
uniform sampler2D sceneTex,aoTex;
uniform bool aoOn;
uniform float aoStrength;
void main() {
    vec3 c=texture(sceneTex,vUV).rgb;
    if(aoOn)c*=mix(1.0,texture(aoTex,vUV).r,aoStrength);
    outColor=vec4(c,1);
}
'''
# -----------------------------------------------------------------------------
# Scene geometry
# -----------------------------------------------------------------------------
CUBE = []

def face(a, b, c, d, n):
    for v in (a, b, c, a, c, d):
        CUBE.extend((*v, *n))

def make_cube():
    p = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
    face(p[1], p[2], p[6], p[5], (1, 0, 0))
    face(p[4], p[7], p[3], p[0], (-1, 0, 0))
    face(p[3], p[7], p[6], p[2], (0, 1, 0))
    face(p[4], p[0], p[1], p[5], (0, -1, 0))
    face(p[5], p[6], p[7], p[4], (0, 0, 1))
    face(p[0], p[3], p[2], p[1], (0, 0, -1))
    return np.asarray(CUBE, np.float32)
SCENE = []

def box(x, y, z, hx, hy, hz, c):
    SCENE.append({'pos': np.array([x, y, z], np.float32), 'half': np.array([hx, hy, hz], np.float32), 'color': np.array(c, np.float32)})

def build_scene():
    SCENE.clear()
    box(0, -0.25, 0, 20.5, 0.25, 20.5, (0.48, 0.5, 0.52))
    wall = (0.52, 0.52, 0.48)
    box(0, 2, -20, 20.5, 2, 0.5, wall)
    box(0, 2, 20, 20.5, 2, 0.5, wall)
    box(-20, 2, 0, 0.5, 2, 20, wall)
    box(20, 2, 0, 0.5, 2, 20, wall)
    crate = (0.66, 0.43, 0.25)
    concrete = (0.47, 0.49, 0.52)
    metal = (0.35, 0.39, 0.43)
    cover = (0.42, 0.45, 0.42)
    box(0, 1.3, -4, 2.6, 1.3, 2.5, concrete)
    box(0, 3.1, -4, 1, 0.5, 1, metal)
    for x, z in [(-8, -8), (-8, -8), (9, -9), (12, 8), (-14, 14), (15, 14), (14, -14), (-14, -14)]:
        box(x, 0.7, z, 0.7, 0.7, 0.7, crate)
    box(-8, 2.1, -8, 0.7, 0.7, 0.7, crate)
    box(9, 2.1, -9, 0.7, 0.7, 0.7, crate)
    box(0, 0.45, 6, 4, 0.45, 0.35, cover)
    box(-6, 0.45, 3, 3, 0.45, 0.35, cover)
    box(7, 0.45, 2, 3, 0.45, 0.35, cover)
    box(0, 0.45, -10, 5, 0.45, 0.35, cover)
    box(-3.5, 1, 10, 1.5, 1, 1.5, concrete)
    box(4, 1, 12, 2, 1, 1, concrete)
    box(-17, 2.5, -3, 0.8, 2.5, 0.8, metal)
    box(17, 2.5, 3, 0.8, 2.5, 0.8, metal)

# -----------------------------------------------------------------------------
# FPS camera, input, collision, and movement
# -----------------------------------------------------------------------------
def front():
    y, p = (math.radians(YAW), math.radians(PITCH))
    return np.array([math.cos(y) * math.cos(p), math.sin(p), math.sin(y) * math.cos(p)], np.float32)

def mouse_cb(win, x, y):
    global FIRST_MOUSE, LAST_X, LAST_Y, YAW, PITCH
    if not LOCKED:
        return
    if FIRST_MOUSE:
        LAST_X, LAST_Y = (x, y)
        FIRST_MOUSE = False
    dx, dy = (x - LAST_X, LAST_Y - y)
    LAST_X, LAST_Y = (x, y)
    YAW += dx * SENS
    PITCH = max(-89, min(89, PITCH + dy * SENS))

def resize_cb(win, w, h):
    global W, H
    W = max(1, w)
    H = max(1, h)
    glViewport(0, 0, W, H)

def collides(px, pz):
    for b in SCENE:
        c, h = (b['pos'], b['half'])
        if px + PLAYER_RADIUS > c[0] - h[0] and px - PLAYER_RADIUS < c[0] + h[0] and (PLAYER[1] < c[1] + h[1]) and (PLAYER[1] + PLAYER_HEIGHT > c[1] - h[1]) and (pz + PLAYER_RADIUS > c[2] - h[2]) and (pz - PLAYER_RADIUS < c[2] + h[2]):
            return True
    return False

def move(win, dt):
    f = front()
    f[1] = 0
    f = norm(f)
    r = norm(np.cross(f, np.array([0, 1, 0], np.float32)))
    v = np.zeros(3, np.float32)
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        v += f
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        v -= f
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        v += r
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        v -= r
    if np.linalg.norm(v) > 1e-06:
        v = norm(v) * SPEED * dt
    nx = PLAYER[0] + v[0]
    if not collides(nx, PLAYER[2]):
        PLAYER[0] = nx
    nz = PLAYER[2] + v[2]
    if not collides(PLAYER[0], nz):
        PLAYER[2] = nz

# -----------------------------------------------------------------------------
# Framebuffer and rendering helpers
# -----------------------------------------------------------------------------
def create_target(w, h, internal, fmt, typ, filtering=GL_NEAREST):
    t = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, t)
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, typ, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return t

def make_fbo(textures, depth=0):
    f = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, f)
    for i, t in enumerate(textures):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, t, 0)
    if depth:
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0)
    if textures:
        glDrawBuffers(len(textures), np.array([GL_COLOR_ATTACHMENT0 + i for i in range(len(textures))], np.uint32))
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError('Framebuffer tamamlanamadi')
    return f

def light_vector():
    az, el = (math.radians(LIGHT_AZ), math.radians(max(1, min(89, LIGHT_EL))))
    return norm(np.array([-math.cos(el) * math.cos(az), -math.sin(el), -math.cos(el) * math.sin(az)], np.float32))

# -----------------------------------------------------------------------------
# Approximate probe-based global illumination
# -----------------------------------------------------------------------------
def bake_gi():
    nx, ny, nz = (8, 3, 8)
    mn = np.array([-19.0, 0.25, -19.0], np.float32)
    mx = np.array([19.0, 5.0, 19.0], np.float32)
    probes = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                probes.append(mn + np.array([x / (nx - 1), y / (ny - 1), z / (nz - 1)], np.float32) * (mx - mn))
    probes = np.array(probes, np.float32)
    rng = np.random.RandomState(7)
    dirs = rng.normal(size=(20, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = np.zeros((len(probes), 3), np.float32)
    L = -light_vector()
    lc = np.array(LIGHT_COLOR, np.float32) * LIGHT_POWER
    for i, p in enumerate(probes):
        acc = np.zeros(3, np.float32)
        count = 0
        for d in dirs:
            best = 1000000000.0
            hit = None
            for b in SCENE:
                c, h = (b['pos'], b['half'])
                tmin = -1000000000.0
                tmax = 1000000000.0
                axis = -1
                sign = 1
                for a in range(3):
                    if abs(float(d[a])) < 1e-07:
                        if p[a] < c[a] - h[a] or p[a] > c[a] + h[a]:
                            tmin = 10000000000.0
                            break
                    else:
                        t1 = (c[a] - h[a] - p[a]) / d[a]
                        t2 = (c[a] + h[a] - p[a]) / d[a]
                        near = min(t1, t2)
                        far = max(t1, t2)
                        if near > tmin:
                            tmin = near
                            axis = a
                            sign = -1 if t1 > t2 else 1
                        tmax = min(tmax, far)
                t = tmin if tmin > 0.001 else tmax
                if t > 0.001 and t < tmax + 0.0001 and (t < best) and (tmax >= max(tmin, 0)):
                    best = t
                    n = np.zeros(3, np.float32)
                    n[axis] = sign
                    hit = (b, n, p + d * t)
            if hit:
                b, n, pt = hit
                nd = max(float(np.dot(n, L)), 0.0)
                acc += b['color'] * lc * (nd / math.pi)
                count += 1
        if count:
            vals[i] = acc / count
    return (mn, mx, (nx, ny, nz), vals)

# -----------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------
def main():
    global LOCKED, FIRST_MOUSE, Q_PREV, SPEED, LIGHT_POWER, LIGHT_AZ, LIGHT_EL, SHADOWS, AO_ON, GI_ON, AO_RADIUS, AO_BIAS, AO_STRENGTH, GI_STRENGTH, GI_BOUNCES
    if not glfw.init():
        raise RuntimeError('GLFW baslatilamadi')
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(W, H, 'Python FPS Test | Shadow + SSAO + Probe GI', None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError('OpenGL 3.3 penceresi olusturulamadi')
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    glfw.set_cursor_pos_callback(win, mouse_cb)
    glfw.set_framebuffer_size_callback(win, resize_cb)
    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)
    imgui.create_context()
    ui = GlfwRenderer(win)
    pmain = program(MAIN_VS, MAIN_FS)
    pdepth = program(DEPTH_VS, DEPTH_FS)
    pao = program(QUAD_VS, SSAO_FS)
    pblur = program(QUAD_VS, BLUR_FS)
    pcomp = program(QUAD_VS, COMP_FS)
    cube = make_cube()
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, cube.nbytes, cube, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glBindVertexArray(0)
    quad = np.array([-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, -1, 0, 0, 1, 1, 1, 1, -1, 1, 0, 1], np.float32)
    qvao = glGenVertexArrays(1)
    qvbo = glGenBuffers(1)
    glBindVertexArray(qvao)
    glBindBuffer(GL_ARRAY_BUFFER, qvbo)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    glBindVertexArray(0)
    build_scene()
    shadow_size = 2048
    shadow = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, shadow)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_size, shadow_size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, np.array([1, 1, 1, 1], np.float32))
    sfbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, sfbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def resize_targets():
        global W, H
        nonlocal gcolor, gnormal, gdepth, gfbo, aotex, aofbo, blurtex, blurfbo
        aw, ah = (max(1, W // 2), max(1, H // 2))
        gcolor = create_target(W, H, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR)
        gnormal = create_target(W, H, GL_RGBA16F, GL_RGBA, GL_FLOAT)
        gdepth = create_target(W, H, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT)
        gfbo = make_fbo([gcolor, gnormal], gdepth)
        aotex = create_target(aw, ah, GL_R16F, GL_RED, GL_FLOAT, GL_LINEAR)
        aofbo = make_fbo([aotex])
        blurtex = create_target(aw, ah, GL_R16F, GL_RED, GL_FLOAT, GL_LINEAR)
        blurfbo = make_fbo([blurtex])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    gcolor = gnormal = gdepth = gfbo = aotex = aofbo = blurtex = blurfbo = 0
    resize_targets()
    last_wh = (W, H)
    rng = np.random.RandomState(5)
    kernel = []
    for i in range(16):
        v = rng.normal(size=3).astype(np.float32)
        v[2] = abs(v[2])
        v = norm(v)
        v *= rng.uniform(0.1, 1.0)
        s = i / 16.0
        v *= 0.1 + 0.9 * s * s
        kernel.append(v)
    kernel = np.ascontiguousarray(kernel, np.float32)
    noise = rng.uniform(-1, 1, (4, 4, 3)).astype(np.float32)
    noise[:, :, 2] = 0
    nt = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, nt)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, noise)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    gi_min, gi_max, gi_grid, gi_vals = bake_gi()
    gtex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, gtex)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F, *gi_grid, 0, GL_RGB, GL_FLOAT, gi_vals.reshape(gi_grid[2], gi_grid[1], gi_grid[0], 3))
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    prev = time.time()
    last_bake_key = None
    while not glfw.window_should_close(win):
        now = time.time()
        dt = min(now - prev, 0.1)
        prev = now
        glfw.poll_events()
        ui.process_inputs()
        q = glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS
        if q and (not Q_PREV):
            LOCKED = not LOCKED
            glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED if LOCKED else glfw.CURSOR_NORMAL)
            FIRST_MOUSE = True
        Q_PREV = q
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(win, True)
        if LOCKED:
            move(win, dt)
        if (W, H) != last_wh:
            resize_targets()
            last_wh = (W, H)
        ld = light_vector()
        cam = PLAYER + np.array([0, EYE, 0], np.float32)
        view = look_at(cam, cam + front(), [0, 1, 0])
        proj = perspective(math.radians(70), W / H, 0.05, 150)
        lightview = look_at(-ld * 45, np.zeros(3, np.float32), [0, 1, 0])
        lightproj = ortho(-32, 32, -32, 32, 1, 120)
        lightspace = lightproj @ lightview
        bake_key = (round(LIGHT_AZ, 1), round(LIGHT_EL, 1), round(LIGHT_POWER, 2), tuple((round(c, 2) for c in LIGHT_COLOR)), GI_BOUNCES)
        if GI_ON and bake_key != last_bake_key:
            gi_min, gi_max, gi_grid, gi_vals = bake_gi()
            glBindTexture(GL_TEXTURE_3D, gtex)
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, *gi_grid, GL_RGB, GL_FLOAT, gi_vals.reshape(gi_grid[2], gi_grid[1], gi_grid[0], 3))
            last_bake_key = bake_key
        glViewport(0, 0, shadow_size, shadow_size)
        glBindFramebuffer(GL_FRAMEBUFFER, sfbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glCullFace(GL_FRONT)
        glUseProgram(pdepth)
        set_mat(pdepth, 'lightSpace', lightspace)
        glBindVertexArray(vao)
        for b in SCENE:
            set_mat(pdepth, 'model', model_mat(b['pos'], b['half']))
            glDrawArrays(GL_TRIANGLES, 0, 36)
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, gfbo)
        glViewport(0, 0, W, H)
        glClearColor(0.42, 0.53, 0.66, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(pmain)
        set_mat(pmain, 'view', view)
        set_mat(pmain, 'projection', proj)
        set_mat(pmain, 'lightSpace', lightspace)
        set3(pmain, 'lightDir', ld)
        set3(pmain, 'lightColor', np.array(LIGHT_COLOR) * LIGHT_POWER)
        set3(pmain, 'ambientColor', AMBIENT)
        set1i(pmain, 'shadowsEnabled', SHADOWS)
        set1i(pmain, 'giEnabled', GI_ON)
        set1f(pmain, 'shadowBias', 0.002)
        set1f(pmain, 'giStrength', GI_STRENGTH)
        set3(pmain, 'giMin', gi_min)
        set3(pmain, 'giMax', gi_max)
        glUniform3iv(glGetUniformLocation(pmain, 'giGrid'), 1, np.array(gi_grid, np.int32))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, shadow)
        set1i(pmain, 'shadowMap', 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, gtex)
        set1i(pmain, 'giTex', 1)
        glBindVertexArray(vao)
        for b in SCENE:
            set_mat(pmain, 'model', model_mat(b['pos'], b['half']))
            set3(pmain, 'objectColor', b['color'])
            glDrawArrays(GL_TRIANGLES, 0, 36)
        aw, ah = (max(1, W // 2), max(1, H // 2))
        if AO_ON:
            glDisable(GL_DEPTH_TEST)
            glBindFramebuffer(GL_FRAMEBUFFER, aofbo)
            glViewport(0, 0, aw, ah)
            glClearColor(1, 1, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(pao)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, gdepth)
            set1i(pao, 'gDepth', 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, gnormal)
            set1i(pao, 'gNormal', 1)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, nt)
            set1i(pao, 'noiseTex', 2)
            set_mat(pao, 'projection', proj)
            set_mat(pao, 'inverseProjection', np.linalg.inv(proj).astype(np.float32))
            set1f(pao, 'radius', AO_RADIUS)
            set1f(pao, 'bias', AO_BIAS)
            glUniform2f(glGetUniformLocation(pao, 'noiseScale'), aw / 4.0, ah / 4.0)
            glUniform3fv(glGetUniformLocation(pao, 'samples'), 16, kernel)
            glBindVertexArray(qvao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindFramebuffer(GL_FRAMEBUFFER, blurfbo)
            glUseProgram(pblur)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, aotex)
            set1i(pblur, 'src', 0)
            glUniform2f(glGetUniformLocation(pblur, 'texel'), 1 / aw, 1 / ah)
            glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, W, H)
        glClearColor(0.05, 0.05, 0.05, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(pcomp)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, gcolor)
        set1i(pcomp, 'sceneTex', 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, blurtex if AO_ON else aotex)
        set1i(pcomp, 'aoTex', 1)
        set1i(pcomp, 'aoOn', AO_ON)
        set1f(pcomp, 'aoStrength', AO_STRENGTH)
        glBindVertexArray(qvao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        imgui.new_frame()
        imgui.set_next_window_size((355, 680), imgui.Cond_.first_use_ever)
        imgui.begin('FPS Test Kontrolleri')
        imgui.text('FPS: %.1f | Fare: %s' % (1 / dt if dt > 0 else 0, 'KILITLI' if LOCKED else 'SERBEST'))
        imgui.text('Q: fare kilidi | ESC: cikis')
        imgui.separator()
        _, SPEED = imgui.slider_float('Hareket hizi', SPEED, 1, 20, '%.1f')
        imgui.separator()
        imgui.text('YONLU ISIK')
        changed, lc = imgui.color_edit3('Isik rengi', LIGHT_COLOR)
        if changed:
            LIGHT_COLOR[:] = [float(x) for x in lc]
        _, LIGHT_POWER = imgui.slider_float('Isik siddeti', LIGHT_POWER, 0, 4, '%.2f')
        _, LIGHT_AZ = imgui.slider_float('Yatay aci', LIGHT_AZ, 0, 360, '%.1f')
        _, LIGHT_EL = imgui.slider_float('Dikey aci', LIGHT_EL, 1, 89, '%.1f')
        changed, ac = imgui.color_edit3('Ortam isigi', AMBIENT)
        if changed:
            AMBIENT[:] = [float(x) for x in ac]
        _, SHADOWS = imgui.checkbox('Golgeler', SHADOWS)
        imgui.separator()
        imgui.text('SSAO')
        _, AO_ON = imgui.checkbox('SSAO etkin', AO_ON)
        _, AO_RADIUS = imgui.slider_float('AO yaricapi', AO_RADIUS, 0.1, 2, '%.2f')
        _, AO_BIAS = imgui.slider_float('AO bias', AO_BIAS, 0.001, 0.1, '%.3f')
        _, AO_STRENGTH = imgui.slider_float('AO gucu', AO_STRENGTH, 0, 1, '%.2f')
        imgui.separator()
        imgui.text('PROBE GI (yaklasik)')
        _, GI_ON = imgui.checkbox('GI etkin', GI_ON)
        _, GI_STRENGTH = imgui.slider_float('GI gucu', GI_STRENGTH, 0, 2, '%.2f')
        imgui.text_wrapped('GI bake isik degisince CPU uzerinde yenilenir; buyuk sahnelerde BVH/GPU bake gerekir.')
        imgui.end()
        imgui.render()
        ui.render(imgui.get_draw_data())
        glfw.swap_buffers(win)
    ui.shutdown()
    glfw.terminate()
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('HATA:', e)
        raise
