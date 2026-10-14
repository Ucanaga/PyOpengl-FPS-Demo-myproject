# -*- coding: utf-8 -*-
"""
FPS test ortamı:
- SSAO (yarı çözünürlük, 16 örnek, tek blur)
- Dinamik GI: irradiance probe grid + 2. derece SH, tek-atımlık (1-2 bounce)
- Klasik shadow map (3x3 PCF)
- ImGui Bundle panel
"""
import sys, math, time, ctypes
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import glfw
from OpenGL.GL import *

# ---------------------------------------------------------------- math
def normalize(v):
    v = np.asarray(v, np.float32); n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def look_at(eye, center, up):
    eye = np.asarray(eye, np.float32); center = np.asarray(center, np.float32)
    up = np.asarray(up, np.float32)
    f = normalize(center - eye); s = normalize(np.cross(f, up)); u = np.cross(s, f)
    m = np.eye(4, np.float32)
    m[0,:3] = s; m[0,3] = -np.dot(s, eye)
    m[1,:3] = u; m[1,3] = -np.dot(u, eye)
    m[2,:3] = -f; m[2,3] = np.dot(f, eye)
    return m

def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy/2.0); m = np.zeros((4,4), np.float32)
    m[0,0] = f/aspect; m[1,1] = f
    m[2,2] = (far+near)/(near-far); m[2,3] = (2*far*near)/(near-far); m[3,2] = -1.0
    return m

def ortho(l,r,b,t,n,f):
    m = np.eye(4, np.float32)
    m[0,0] = 2/(r-l); m[1,1] = 2/(t-b); m[2,2] = -2/(f-n)
    m[0,3] = -(r+l)/(r-l); m[1,3] = -(t+b)/(t-b); m[2,3] = -(f+n)/(f-n)
    return m

def model_matrix(pos, half):
    m = np.eye(4, np.float32)
    m[0,0] = half[0]*2; m[1,1] = half[1]*2; m[2,2] = half[2]*2
    m[0,3], m[1,3], m[2,3] = pos[0], pos[1], pos[2]
    return m

# ---------------------------------------------------------------- shaders
def compile_shader(src, typ):
    s = glCreateShader(typ); glShaderSource(s, src); glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError("Shader:\n" + glGetShaderInfoLog(s).decode())
    return s

def create_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    p = glCreateProgram(); glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError("Link:\n" + glGetProgramInfoLog(p).decode())
    glDeleteShader(vs); glDeleteShader(fs); return p

MAIN_VS = """
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 model, view, projection, lightSpaceMatrix;
out vec3 FragPos; out vec3 Normal; out vec4 FragPosLightSpace;
void main(){
    vec4 w = model * vec4(aPos,1.0);
    FragPos = w.xyz;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    FragPosLightSpace = lightSpaceMatrix * w;
    gl_Position = projection * view * w;
}
"""

MAIN_FS = """
#version 430 core
in vec3 FragPos; in vec3 Normal; in vec4 FragPosLightSpace;
layout(location=0) out vec4 FragColor;
layout(location=1) out vec4 OutNormal;

uniform vec3 lightDir, lightColor, ambientColor, viewPos, objectColor;
uniform sampler2D shadowMap;
uniform bool shadowsEnabled;
uniform float shadowBias;
uniform mat4 view;

// --- Irradiance probe grid SH via SSBO ---
layout(std430, binding=0) buffer SHBuffer { float sh_data[]; };
uniform ivec3 gridSize;   // (NX, NY, NZ) = (8, 4, 8)
uniform vec3 giMin, giMax;
uniform bool giEnabled;
uniform float giIntensity;

float shadow_factor(vec3 N, vec3 L){
    vec3 pc = FragPosLightSpace.xyz / FragPosLightSpace.w;
    pc = pc * 0.5 + 0.5;
    if (pc.z > 1.0) return 0.0;
    float cur = pc.z;
    float bias = max(shadowBias*(1.0-dot(N,L)), shadowBias*0.3);
    float sh = 0.0;
    vec2 ts = vec2(1.0)/vec2(textureSize(shadowMap,0));
    for (int x=-1;x<=1;++x) for (int y=-1;y<=1;++y){
        float pd = texture(shadowMap, pc.xy+vec2(x,y)*ts).r;
        sh += (cur - bias > pd) ? 1.0 : 0.0;
    }
    return sh/9.0;
}

// Tek koef için 8-köşe trilinear (SH buffer küçük → cache'de kalır)
vec3 trilinear_coeff(int coeff, ivec3 base, vec3 w){
    vec3 r = vec3(0.0);
    for (int k=0;k<8;++k){
        int dx = k & 1;
        int dy = (k >> 1) & 1;
        int dz = (k >> 2) & 1;
        ivec3 g = base + ivec3(dx,dy,dz);
        int pidx = g.x + g.y*gridSize.x + g.z*gridSize.x*gridSize.y;
        int off = pidx * 27 + coeff * 3;
        vec3 v = vec3(sh_data[off], sh_data[off+1], sh_data[off+2]);
        float ww = (dx>0 ? w.x : 1.0-w.x)
                 * (dy>0 ? w.y : 1.0-w.y)
                 * (dz>0 ? w.z : 1.0-w.z);
        r += ww * v;
    }
    return r;
}

vec3 eval_irradiance(vec3 N){
    vec3 uvw = clamp((FragPos - giMin)/(giMax - giMin), 0.0, 1.0);
    vec3 p = uvw * vec3(gridSize) - 0.5;
    ivec3 i0 = ivec3(floor(p));
    vec3 f = clamp(p - vec3(i0), 0.0, 1.0);
    i0 = clamp(i0, ivec3(0), gridSize - ivec3(2));

    vec3 c[9];
    for (int j=0;j<9;++j) c[j] = trilinear_coeff(j, i0, f);

    const float Y0 = 0.282095;
    const float Y1 = 0.488603;
    const float Y2 = 1.092548;
    const float Y3 = 0.315392;
    const float Y4 = 0.546274;
    const float PI = 3.14159265;

    vec3 E = c[0] * (PI * Y0);
    E += c[1] * ((2.0*PI/3.0) * Y1 * N.y);
    E += c[2] * ((2.0*PI/3.0) * Y1 * N.z);
    E += c[3] * ((2.0*PI/3.0) * Y1 * N.x);
    E += c[4] * ((PI/4.0) * Y2 * N.x * N.y);
    E += c[5] * ((PI/4.0) * Y2 * N.y * N.z);
    E += c[6] * ((PI/4.0) * Y3 * (3.0*N.z*N.z - 1.0));
    E += c[7] * ((PI/4.0) * Y2 * N.x * N.z);
    E += c[8] * ((PI/4.0) * Y4 * (N.x*N.x - N.y*N.y));
    return max(E, 0.0);
}

void main(){
    vec3 N = normalize(Normal);
    vec3 L = normalize(-lightDir);
    float diff = max(dot(N,L), 0.0);
    vec3 ambient = ambientColor * objectColor;
    vec3 diffuse = diff * lightColor * objectColor;
    float shadow = shadowsEnabled ? shadow_factor(N,L) : 0.0;
    vec3 color = ambient + (1.0 - shadow) * diffuse;

    if (giEnabled) {
        vec3 gi = eval_irradiance(N);
        color += gi * objectColor * giIntensity / 3.14159265;
    }
    FragColor = vec4(color, 1.0);
    OutNormal = vec4(normalize(mat3(view) * N), 1.0);
}
"""

DEPTH_VS = """
#version 430 core
layout(location=0) in vec3 aPos;
uniform mat4 model, lightSpaceMatrix;
void main(){ gl_Position = lightSpaceMatrix * model * vec4(aPos,1.0); }
"""
DEPTH_FS = "#version 430 core\nvoid main(){}"

FULLSCREEN_VS = """
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 TexCoord;
void main(){ TexCoord = aUV; gl_Position = vec4(aPos,0.0,1.0); }
"""

SSAO_FS = """
#version 430 core
in vec2 TexCoord;
out float FragAO;
uniform sampler2D gNormal, gDepth, noiseTex;
uniform mat4 projection, inverseProjection;
uniform vec3 samples[16];
uniform float radius, bias, power;
uniform vec2 aoSize;

vec3 viewPosFromDepth(vec2 uv, float d){
    vec4 clip = vec4(uv*2.0-1.0, d*2.0-1.0, 1.0);
    vec4 v = inverseProjection * clip;
    return v.xyz / v.w;
}
void main(){
    // G-buffer tam çözünürlüklü ama UV [0,1] paylaşıldığı için doğrudan örnekleme geçerli
    float d = texture(gDepth, TexCoord).r;
    if (d >= 1.0) { FragAO = 1.0; return; }
    vec3 n = texture(gNormal, TexCoord).xyz;
    if (dot(n,n) < 1e-4) { FragAO = 1.0; return; }
    n = normalize(n);

    vec3 fragPos = viewPosFromDepth(TexCoord, d);

    vec3 rnd = texture(noiseTex, TexCoord * aoSize / 4.0).xyz;
    rnd.z = 0.0;
    vec3 t = rnd - n * dot(rnd, n);
    if (dot(t,t) < 1e-6) t = vec3(1,0,0);
    t = normalize(t);
    mat3 TBN = mat3(t, cross(n,t), n);

    float occ = 0.0;
    for (int i=0;i<16;++i){
        vec3 sp = fragPos + TBN * samples[i] * radius;
        vec4 off = projection * vec4(sp,1.0);
        off.xyz /= off.w; off.xyz = off.xyz*0.5 + 0.5;
        if (off.x<0.0||off.x>1.0||off.y<0.0||off.y>1.0) continue;
        float sd = texture(gDepth, off.xy).r;
        vec3 svp = viewPosFromDepth(off.xy, sd);
        float rc = smoothstep(0.0, 1.0, radius / max(1e-4, abs(fragPos.z - svp.z)));
        occ += (svp.z >= sp.z + bias ? 1.0 : 0.0) * rc;
    }
    occ = 1.0 - occ / 16.0;
    FragAO = pow(clamp(occ, 0.0, 1.0), power);
}
"""

BLUR_FS = """
#version 430 core
in vec2 TexCoord;
out float FragAO;
uniform sampler2D aoTex;
uniform vec2 texelSize;
void main(){
    float s = 0.0;
    for (int x=-2;x<=2;++x) for (int y=-2;y<=2;++y)
        s += texture(aoTex, TexCoord + vec2(x,y)*texelSize).r;
    FragAO = s/25.0;
}
"""

COMPOSITE_FS = """
#version 430 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D sceneColor, aoTex;
uniform bool useAO;
uniform float aoStrength;
void main(){
    vec3 col = texture(sceneColor, TexCoord).rgb;
    if (useAO) {
        float ao = texture(aoTex, TexCoord).r;
        // AO'yu doğrudan tümden karartmak yerine yumuşak blend
        col *= mix(1.0, ao, aoStrength);
    }
    FragColor = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------- geometry
CUBE_VERTS = np.array([
     0.5,-0.5,-0.5, 1,0,0,  0.5, 0.5,-0.5, 1,0,0,  0.5, 0.5, 0.5, 1,0,0,
     0.5,-0.5,-0.5, 1,0,0,  0.5, 0.5, 0.5, 1,0,0,  0.5,-0.5, 0.5, 1,0,0,
    -0.5,-0.5, 0.5,-1,0,0, -0.5, 0.5, 0.5,-1,0,0, -0.5, 0.5,-0.5,-1,0,0,
    -0.5,-0.5, 0.5,-1,0,0, -0.5, 0.5,-0.5,-1,0,0, -0.5,-0.5,-0.5,-1,0,0,
    -0.5, 0.5,-0.5, 0,1,0, -0.5, 0.5, 0.5, 0,1,0,  0.5, 0.5, 0.5, 0,1,0,
    -0.5, 0.5,-0.5, 0,1,0,  0.5, 0.5, 0.5, 0,1,0,  0.5, 0.5,-0.5, 0,1,0,
    -0.5,-0.5, 0.5, 0,-1,0,-0.5,-0.5,-0.5, 0,-1,0, 0.5,-0.5,-0.5, 0,-1,0,
    -0.5,-0.5, 0.5, 0,-1,0, 0.5,-0.5,-0.5, 0,-1,0, 0.5,-0.5, 0.5, 0,-1,0,
    -0.5,-0.5, 0.5, 0,0,1,  0.5,-0.5, 0.5, 0,0,1,  0.5, 0.5, 0.5, 0,0,1,
    -0.5,-0.5, 0.5, 0,0,1,  0.5, 0.5, 0.5, 0,0,1, -0.5, 0.5, 0.5, 0,0,1,
     0.5,-0.5,-0.5, 0,0,-1,-0.5,-0.5,-0.5, 0,0,-1,-0.5, 0.5,-0.5, 0,0,-1,
     0.5,-0.5,-0.5, 0,0,-1,-0.5, 0.5,-0.5, 0,0,-1, 0.5, 0.5,-0.5, 0,0,-1,
], dtype=np.float32)

QUAD_VERTS = np.array([
    -1,-1,0,0,  1,-1,1,0,  1,1,1,1,
    -1,-1,0,0,  1,1,1,1,  -1,1,0,1,
], dtype=np.float32)

# ---------------------------------------------------------------- scene
SCENE = []
def add_box(cx,cy,cz,hx,hy,hz,color):
    SCENE.append({'pos':(cx,cy,cz),'half':(hx,hy,hz),'color':color})

def build_scene():
    SCENE.clear()
    add_box(0,-0.25,0, 20.5,0.25,20.5, (0.55,0.55,0.55))
    wall=(0.62,0.58,0.50)
    add_box(0,2,-20.5, 20.5,2,0.5, wall)
    add_box(0,2, 20.5, 20.5,2,0.5, wall)
    add_box(-20.5,2,0, 0.5,2,20.5, wall)
    add_box( 20.5,2,0, 0.5,2,20.5, wall)
    crate=(0.78,0.58,0.32); crate2=(0.72,0.52,0.28)
    concrete=(0.62,0.62,0.66); metal=(0.45,0.48,0.55); barrier=(0.60,0.62,0.58)
    add_box(0,1.5,-4, 3,1.5,3, concrete)
    add_box(0,3.4,-4, 1.2,0.4,1.2, metal)
    add_box(-8,1,-8, 1,1,1, crate); add_box(-8,3,-8, 1,1,1, crate2)
    add_box(-6,0.75,-9, 0.75,0.75,0.75, crate)
    add_box(9,1,-9, 1,1,1, crate); add_box(9,3,-9, 1,1,1, crate)
    add_box(11,0.5,-7, 0.5,0.5,0.5, crate2)
    add_box(-12,0.5,6, 0.5,0.5,0.5, crate2)
    add_box(-12,1.5,6, 0.5,0.5,0.5, crate2)
    add_box(-10.5,0.5,7, 0.5,0.5,0.5, crate2)
    add_box(12,1.5,8, 1.5,1.5,1.5, crate)
    add_box(0,0.45, 6, 4,0.45,0.35, barrier)
    add_box(-6,0.45, 3, 3,0.45,0.35, barrier)
    add_box( 7,0.45, 2, 3,0.45,0.35, barrier)
    add_box(0,0.45,-10, 5,0.45,0.35, barrier)
    add_box(-3.5,1,10, 1.5,1,1.5, concrete)
    add_box(4,1,12, 2,1,1, concrete)
    add_box(-15,1.25,-14, 2,1.25,2, concrete)
    add_box(15,1.25,14, 2,1.25,2, concrete)
    add_box(14,0.75,-14, 1.5,0.75,1.5, concrete)
    add_box(-14,0.75,14, 1.5,0.75,1.5, concrete)
    add_box(-17,3,-3, 1,3,1, metal)
    add_box(17,3,3, 1,3,1, metal)

# ---------------------------------------------------------------- globals
WIDTH, HEIGHT = 1280, 720
mouse_locked = True; first_mouse = True
last_x, last_y = WIDTH/2, HEIGHT/2
cam_yaw = -90.0; cam_pitch = 0.0; mouse_sensitivity = 0.12
player_pos = np.array([0,0,15], np.float32)
player_radius = 0.4; player_height = 1.7; eye_height = 1.6
move_speed = 6.0; q_prev = False

light_color = [1.0, 0.97, 0.9]
light_intensity = 1.0
light_azimuth = 45.0; light_elevation = 55.0
ambient_color = [0.22,0.24,0.28]
shadows_enabled = True; shadow_bias = 0.0025
ssao_enabled = True; ssao_radius = 0.5; ssao_bias = 0.02
ssao_power = 1.6; ao_strength = 0.85
gi_enabled = True; gi_intensity = 0.9
gi_bounces = 2

PROBE_MIN = np.array([-20.0, 0.6, -20.0], np.float32)
PROBE_MAX = np.array([ 20.0, 5.0,  20.0], np.float32)
PROBE_NX, PROBE_NY, PROBE_NZ = 8, 4, 8
N_RAYS = 48

# ---------------------------------------------------------------- collision
def collides(px, pz):
    pminx, pmaxx = px - player_radius, px + player_radius
    pminy, pmaxy = player_pos[1], player_pos[1] + player_height
    pminz, pmaxz = pz - player_radius, pz + player_radius
    for b in SCENE:
        cx,cy,cz = b['pos']; hx,hy,hz = b['half']
        if (pminx < cx+hx and pmaxx > cx-hx and pminy < cy+hy and pmaxy > cy-hy
            and pminz < cz+hz and pmaxz > cz-hz):
            return True
    return False

# ---------------------------------------------------------------- camera
def cam_front():
    y = math.radians(cam_yaw); p = math.radians(cam_pitch)
    return np.array([math.cos(y)*math.cos(p), math.sin(p), math.sin(y)*math.cos(p)], np.float32)

def mouse_callback(window, xpos, ypos):
    global first_mouse, last_x, last_y, cam_yaw, cam_pitch
    if not mouse_locked: return
    if first_mouse:
        last_x, last_y = xpos, ypos; first_mouse = False
    dx = xpos - last_x; dy = last_y - ypos
    last_x, last_y = xpos, ypos
    cam_yaw += dx * mouse_sensitivity
    cam_pitch = max(-89.0, min(89.0, cam_pitch + dy * mouse_sensitivity))

def fb_cb(window, w, h):
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = w, h
    glViewport(0,0,w,h)

def process_movement(dt):
    front = cam_front(); front[1] = 0.0
    if np.linalg.norm(front) > 1e-6: front = normalize(front)
    right = normalize(np.cross(front, np.array([0,1,0], np.float32)))
    vel = np.zeros(3, np.float32)
    ctx = glfw.get_current_context()
    if glfw.get_key(ctx, glfw.KEY_W) == glfw.PRESS: vel += front
    if glfw.get_key(ctx, glfw.KEY_S) == glfw.PRESS: vel -= front
    if glfw.get_key(ctx, glfw.KEY_D) == glfw.PRESS: vel += right
    if glfw.get_key(ctx, glfw.KEY_A) == glfw.PRESS: vel -= right
    if np.linalg.norm(vel) > 1e-6:
        vel = normalize(vel) * move_speed * dt
    nx = player_pos[0] + float(vel[0])
    if not collides(nx, player_pos[2]): player_pos[0] = nx
    nz = player_pos[2] + float(vel[2])
    if not collides(player_pos[0], nz): player_pos[2] = nz

# ---------------------------------------------------------------- triangles
def build_triangles(scene):
    tris = []; albedos = []
    face_idx = [[0,1,3,2],[4,5,7,6],[0,2,6,4],[1,5,7,3],[2,3,7,6],[0,4,5,1]]
    for b in scene:
        c = np.array(b['pos'], np.float32); h = np.array(b['half'], np.float32)
        col = np.array(b['color'], np.float32)
        corners = np.array([
            [c[0]-h[0],c[1]-h[1],c[2]-h[2]], [c[0]+h[0],c[1]-h[1],c[2]-h[2]],
            [c[0]-h[0],c[1]+h[1],c[2]-h[2]], [c[0]+h[0],c[1]+h[1],c[2]-h[2]],
            [c[0]-h[0],c[1]-h[1],c[2]+h[2]], [c[0]+h[0],c[1]-h[1],c[2]+h[2]],
            [c[0]-h[0],c[1]+h[1],c[2]+h[2]], [c[0]+h[0],c[1]+h[1],c[2]+h[2]],
        ], np.float32)
        for q in face_idx:
            a,bq,cc,dq = corners[q[0]], corners[q[1]], corners[q[2]], corners[q[3]]
            tris.append([a,bq,cc]); albedos.append(col)
            tris.append([a,cc,dq]); albedos.append(col)
    return np.array(tris, np.float32), np.array(albedos, np.float32)

def ray_cast_all(origins, dirs, tris):
    N = origins.shape[0]; T = tris.shape[0]
    hit_t = np.full(N, np.inf, np.float32)
    hit_idx = np.full(N, -1, np.int32)
    eps = 1e-6
    for i in range(T):
        V0, V1, V2 = tris[i]
        E1 = V1-V0; E2 = V2-V0
        Pv = np.cross(dirs, E2)
        det = np.einsum('ij,ij->i', E1[None,:], Pv)
        vd = np.abs(det) > eps
        inv = np.where(vd, 1.0/np.where(vd, det, 1.0), 0.0)
        Tv = origins - V0
        U = np.einsum('ij,ij->i', Tv, Pv) * inv
        Qv = np.cross(Tv, E1)
        V = np.einsum('ij,ij->i', dirs, Qv) * inv
        t = np.einsum('ij,ij->i', E2[None,:], Qv) * inv
        hit = vd & (U >= 0) & (V >= 0) & (U+V <= 1) & (t > eps)
        closer = hit & (t < hit_t)
        hit_t[closer] = t[closer]; hit_idx[closer] = i
    return hit_idx, hit_t

def fibonacci_sphere(n):
    pts = np.zeros((n,3), np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i/(n-1))*2.0 if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y*y))
        th = phi*i
        pts[i] = [math.cos(th)*r, y, math.sin(th)*r]
    return pts

def sh_basis(dirs):
    x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
    Y = np.zeros((dirs.shape[0], 9), np.float32)
    Y[:,0] = 0.282095
    Y[:,1] = 0.488603 * y
    Y[:,2] = 0.488603 * z
    Y[:,3] = 0.488603 * x
    Y[:,4] = 1.092548 * x * y
    Y[:,5] = 1.092548 * y * z
    Y[:,6] = 0.315392 * (3*z*z - 1.0)
    Y[:,7] = 1.092548 * x * z
    Y[:,8] = 0.546274 * (x*x - y*y)
    return Y

# ---------------------------------------------------------------- bake
def build_probes():
    probes = []
    for iz in range(PROBE_NZ):
        for iy in range(PROBE_NY):
            for ix in range(PROBE_NX):
                t = np.array([(ix+0.5)/PROBE_NX, (iy+0.5)/PROBE_NY, (iz+0.5)/PROBE_NZ], np.float32)
                probes.append(PROBE_MIN + t*(PROBE_MAX - PROBE_MIN))
    return np.array(probes, np.float32)

def bake_probes(scene, probes):
    """Ray-cast + nearest probe lookup + SH basis önbellek."""
    tris, albedos = build_triangles(scene)
    P = probes.shape[0]; M = N_RAYS

    # Geçerlilik: prob kutuların içindeyse geçersiz
    valid_probe = np.ones(P, bool)
    for b in scene:
        c = np.array(b['pos'], np.float32)
        h = np.array(b['half'], np.float32)
        inside = np.all(np.abs(probes - c) < h, axis=1)
        valid_probe &= ~inside

    dirs = fibonacci_sphere(M)
    Y = sh_basis(dirs).astype(np.float32)
    w = 4.0 * math.pi / M

    O = np.repeat(probes, M, axis=0)
    D = np.tile(dirs, (P,1))
    print("GI bake: %d probe x %d ray = %d ray" % (P, M, P*M))
    hit_idx, hit_t = ray_cast_all(O, D, tris)
    valid_hit = hit_idx >= 0
    print("  %d/%d isabet" % (int(valid_hit.sum()), P*M))

    safe = np.where(valid_hit, hit_idx, 0)
    V0 = tris[safe,0]; V1 = tris[safe,1]; V2 = tris[safe,2]
    n = np.cross(V1-V0, V2-V0)
    ln = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    n = n / ln
    # Yüzü bize dönük yap
    facing = np.einsum('ij,ij->i', n, D)
    n[facing > 0] *= -1.0
    N_H = n.astype(np.float32)
    A_H = albedos[safe].astype(np.float32)
    N_H[~valid_hit] = 0; A_H[~valid_hit] = 0

    # Nearest geçerli prob
    hit_pts = O + D * hit_t[:,None]
    query = hit_pts + N_H * 0.15
    vp_pts = probes[valid_probe]
    vp_idx = np.where(valid_probe)[0]
    d2 = np.sum((query[:,None,:] - vp_pts[None,:,:])**2, axis=2)
    nearest_v = np.argmin(d2, axis=1)
    lookup = vp_idx[nearest_v].astype(np.int32)

    YN = sh_basis(N_H).astype(np.float32)

    return {
        'P': P, 'M': M, 'w': w, 'Y': Y,
        'N_H': N_H.reshape(P,M,3),
        'A_H': A_H.reshape(P,M,3),
        'YN': YN.reshape(P,M,9),
        'lookup': lookup.reshape(P,M),
        'valid_hit': valid_hit.reshape(P,M),
        'valid_probe': valid_probe,
    }

def compute_sh(bake, light_dir, light_col, n_bounces):
    """Doğrudan + N bounce SH. Döner: (P,9,3)."""
    P = bake['P']; M = bake['M']; w = bake['w']
    Y = bake['Y']; N_H = bake['N_H']; A_H = bake['A_H']
    YN = bake['YN']; lookup = bake['lookup']; valid_hit = bake['valid_hit']
    valid_probe = bake['valid_probe']

    L = -np.asarray(light_dir, np.float32)
    lc = np.asarray(light_col, np.float32)

    # Doğrudan ışık -> yüzeydeki irradiance
    ndotl = np.clip(np.einsum('pmc,c->pm', N_H, L), 0.0, None)
    E_dir = ndotl[..., None] * lc[None,None,:]
    L_emit = A_H * E_dir / math.pi * valid_hit[..., None]
    probe_SH = np.einsum('mj,m,pmc->pjc', Y, w, L_emit).astype(np.float32)

    total = probe_SH.copy()
    kcoef = np.array([math.pi, 2*math.pi/3, 2*math.pi/3, 2*math.pi/3,
                      math.pi/4, math.pi/4, math.pi/4, math.pi/4, math.pi/4], np.float32)

    for _ in range(n_bounces):
        sh_at_hit = probe_SH[lookup]  # (P,M,9,3)
        E_ind = np.einsum('pmj,j,pmjc->pmc', YN, kcoef, sh_at_hit)
        L_emit = A_H * np.maximum(E_ind, 0.0) / math.pi * valid_hit[..., None]
        probe_SH_new = np.einsum('mj,m,pmc->pjc', Y, w, L_emit).astype(np.float32)
        total += probe_SH_new
        probe_SH = probe_SH_new
        # Geçersiz probe'ları sıfırla (sızıntıyı önler)
        probe_SH[~valid_probe] = 0

    total[~valid_probe] = 0
    return np.ascontiguousarray(total, np.float32)

# ---------------------------------------------------------------- RT
class RenderTargets:
    def __init__(self):
        self.w = 0; self.h = 0
        self.ao_div = 2; self.shadow_size = 2048
        self.depth_tex = 0; self.shadow_fbo = 0
        self.gbuf_color = 0; self.gbuf_normal = 0; self.gbuf_depth = 0; self.gbuf_fbo = 0
        self.ao_tex = 0; self.ao_fbo = 0
        self.ao_blur_tex = 0; self.ao_blur_fbo = 0
        self.noise_tex = 0
        self._static()

    def _static(self):
        self.depth_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, self.shadow_size, self.shadow_size,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                         np.array([1,1,1,1], np.float32))
        self.shadow_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_tex, 0)
        glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE)

        rng = np.random.RandomState(0)
        nd = np.zeros((4,4,3), np.float32)
        nd[:,:,0] = rng.uniform(-1,1,(4,4)); nd[:,:,1] = rng.uniform(-1,1,(4,4))
        self.noise_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.noise_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, nd.tobytes())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _mk(self, h, internal, w, hh, fmt, typ, filt):
        if h[0] == 0: h[0] = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, h[0])
        glTexImage2D(GL_TEXTURE_2D, 0, internal, w, hh, 0, fmt, typ, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filt)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filt)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def ensure_size(self, w, h):
        if w == self.w and h == self.h: return
        self.w, self.h = w, h
        aw = max(1, w // self.ao_div); ah = max(1, h // self.ao_div)

        hd = [self.gbuf_color];  self._mk(hd, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, GL_LINEAR);  self.gbuf_color = hd[0]
        hd = [self.gbuf_normal]; self._mk(hd, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, GL_NEAREST); self.gbuf_normal = hd[0]
        hd = [self.gbuf_depth];  self._mk(hd, GL_DEPTH_COMPONENT32F, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, GL_NEAREST); self.gbuf_depth = hd[0]

        if self.gbuf_fbo == 0: self.gbuf_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.gbuf_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.gbuf_color, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.gbuf_normal, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_TEXTURE_2D, self.gbuf_depth, 0)
        glDrawBuffers(2, np.array([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1], np.uint32))
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("G-buffer FBO")

        hd = [self.ao_tex]; self._mk(hd, GL_R8, aw, ah, GL_RED, GL_UNSIGNED_BYTE, GL_LINEAR); self.ao_tex = hd[0]
        if self.ao_fbo == 0: self.ao_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.ao_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.ao_tex, 0)
        glDrawBuffers(1, np.array([GL_COLOR_ATTACHMENT0], np.uint32))
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("AO FBO")

        hd = [self.ao_blur_tex]; self._mk(hd, GL_R8, aw, ah, GL_RED, GL_UNSIGNED_BYTE, GL_LINEAR); self.ao_blur_tex = hd[0]
        if self.ao_blur_fbo == 0: self.ao_blur_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.ao_blur_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.ao_blur_tex, 0)
        glDrawBuffers(1, np.array([GL_COLOR_ATTACHMENT0], np.uint32))
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("AO blur FBO")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

# ---------------------------------------------------------------- main
def main():
    global mouse_locked, first_mouse, q_prev, WIDTH, HEIGHT
    global move_speed
    global light_intensity, light_azimuth, light_elevation
    global shadows_enabled, shadow_bias
    global ssao_enabled, ssao_radius, ssao_bias, ssao_power, ao_strength
    global gi_enabled, gi_intensity, gi_bounces

    if not glfw.init():
        print("GLFW yok"); sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIDTH, HEIGHT, "FPS Test - SSAO + SH Probe GI", None, None)
    if not window:
        glfw.terminate(); print("Pencere yok"); sys.exit(1)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_framebuffer_size_callback(window, fb_cb)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    imgui.create_context()
    impl = GlfwRenderer(window)

    main_prog = create_program(MAIN_VS, MAIN_FS)
    depth_prog = create_program(DEPTH_VS, DEPTH_FS)
    ssao_prog = create_program(FULLSCREEN_VS, SSAO_FS)
    blur_prog = create_program(FULLSCREEN_VS, BLUR_FS)
    comp_prog = create_program(FULLSCREEN_VS, COMPOSITE_FS)

    vao = glGenVertexArrays(1); vbo = glGenBuffers(1)
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, CUBE_VERTS.nbytes, CUBE_VERTS, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,24,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,24,ctypes.c_void_p(12))
    glBindVertexArray(0)

    qvao = glGenVertexArrays(1); qvbo = glGenBuffers(1)
    glBindVertexArray(qvao); glBindBuffer(GL_ARRAY_BUFFER, qvbo)
    glBufferData(GL_ARRAY_BUFFER, QUAD_VERTS.nbytes, QUAD_VERTS, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(0))
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,ctypes.c_void_p(8))
    glBindVertexArray(0)

    RT = RenderTargets(); RT.ensure_size(WIDTH, HEIGHT)

    rng = np.random.RandomState(42)
    kernel = np.zeros((16,3), np.float32)
    for i in range(16):
        v = normalize(rng.normal(size=3).astype(np.float32))
        v *= rng.uniform(0.1, 1.0)
        s = i/16.0; s = 0.1 + 0.9*s*s
        kernel[i] = v * s
    kernel = np.ascontiguousarray(kernel, np.float32)

    build_scene()
    probes = build_probes()
    bake = bake_probes(SCENE, probes)
    print("GI bake tamam.")

    P = bake['P']
    sh_size = P * 9 * 3 * 4
    sh_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sh_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, sh_size, None, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sh_ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    U = lambda p, n: glGetUniformLocation(p, n)
    ssao_loc = {k: U(ssao_prog, k) for k in
                ['gNormal','gDepth','noiseTex','projection','inverseProjection',
                 'samples','radius','bias','power','aoSize']}
    blur_loc = {k: U(blur_prog, k) for k in ['aoTex','texelSize']}
    comp_loc = {k: U(comp_prog, k) for k in ['sceneColor','aoTex','useAO','aoStrength']}

    glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glCullFace(GL_BACK)

    prev = time.time()
    last_light_key = None

    while not glfw.window_should_close(window):
        now = time.time(); dt = min(now - prev, 0.1); prev = now
        glfw.poll_events(); impl.process_inputs()

        q = glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS
        if q and not q_prev:
            mouse_locked = not mouse_locked
            if mouse_locked:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
                first_mouse = True
            else:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        q_prev = q
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        process_movement(dt)
        RT.ensure_size(WIDTH, HEIGHT)

        front = cam_front()
        cam_pos = player_pos + np.array([0, eye_height, 0], np.float32)
        view = look_at(cam_pos, cam_pos + front, (0,1,0))
        proj = perspective(math.radians(70.0), max(1,WIDTH)/max(1,HEIGHT), 0.05, 200.0)
        inv_proj = np.ascontiguousarray(np.linalg.inv(proj.astype(np.float64)).astype(np.float32))

        az = math.radians(light_azimuth); el = math.radians(max(1.0, min(89.0, light_elevation)))
        light_dir = np.array([-math.cos(el)*math.cos(az), -math.sin(el),
                              -math.cos(el)*math.sin(az)], np.float32)
        light_col = np.array(light_color, np.float32) * light_intensity

        lv = look_at(-light_dir*45.0, np.zeros(3, np.float32), (0,1,0))
        lproj = ortho(-32,32,-32,32,1,120)
        light_space = lproj @ lv

        # SH güncelle: ışık değişince veya ilk frame
        key = (round(light_azimuth,2), round(light_elevation,2),
               round(light_intensity,3), tuple(round(c,3) for c in light_color),
               gi_bounces)
        if gi_enabled and key != last_light_key:
            sh = compute_sh(bake, light_dir, light_col, gi_bounces)  # (P,9,3)
            data = sh.reshape(-1).astype(np.float32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sh_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            last_light_key = key

        # Shadow pass
        glViewport(0,0,RT.shadow_size,RT.shadow_size)
        glBindFramebuffer(GL_FRAMEBUFFER, RT.shadow_fbo)
        glClear(GL_DEPTH_BUFFER_BIT); glCullFace(GL_FRONT)
        glUseProgram(depth_prog)
        glUniformMatrix4fv(U(depth_prog,"lightSpaceMatrix"),1,GL_TRUE,light_space)
        glBindVertexArray(vao)
        for b in SCENE:
            glUniformMatrix4fv(U(depth_prog,"model"),1,GL_TRUE,model_matrix(b['pos'],b['half']))
            glDrawArrays(GL_TRIANGLES,0,36)
        glBindVertexArray(0); glCullFace(GL_BACK)

        # Main -> G-buffer
        glBindFramebuffer(GL_FRAMEBUFFER, RT.gbuf_fbo)
        glViewport(0,0,WIDTH,HEIGHT)
        glClearColor(0.45, 0.55, 0.68, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(main_prog)
        glUniformMatrix4fv(U(main_prog,"view"),1,GL_TRUE,view)
        glUniformMatrix4fv(U(main_prog,"projection"),1,GL_TRUE,proj)
        glUniformMatrix4fv(U(main_prog,"lightSpaceMatrix"),1,GL_TRUE,light_space)
        glUniform3fv(U(main_prog,"lightDir"),1,light_dir)
        glUniform3f(U(main_prog,"lightColor"), *light_col)
        glUniform3f(U(main_prog,"ambientColor"), *ambient_color)
        glUniform3f(U(main_prog,"viewPos"), *cam_pos)
        glUniform1i(U(main_prog,"shadowsEnabled"), 1 if shadows_enabled else 0)
        glUniform1f(U(main_prog,"shadowBias"), float(shadow_bias))
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, RT.depth_tex)
        glUniform1i(U(main_prog,"shadowMap"), 0)

        glUniform3iv(U(main_prog,"gridSize"), 1, np.array([PROBE_NX, PROBE_NY, PROBE_NZ], np.int32))
        glUniform3f(U(main_prog,"giMin"), *PROBE_MIN)
        glUniform3f(U(main_prog,"giMax"), *PROBE_MAX)
        glUniform1i(U(main_prog,"giEnabled"), 1 if gi_enabled else 0)
        glUniform1f(U(main_prog,"giIntensity"), float(gi_intensity))

        glBindVertexArray(vao)
        for b in SCENE:
            glUniformMatrix4fv(U(main_prog,"model"),1,GL_TRUE,model_matrix(b['pos'],b['half']))
            glUniform3f(U(main_prog,"objectColor"), *b['color'])
            glDrawArrays(GL_TRIANGLES,0,36)
        glBindVertexArray(0)

        # SSAO
        if ssao_enabled:
            aw = max(1, WIDTH // RT.ao_div); ah = max(1, HEIGHT // RT.ao_div)
            glBindFramebuffer(GL_FRAMEBUFFER, RT.ao_fbo)
            glViewport(0,0,aw,ah)
            glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND)
            glClearColor(1,1,1,1); glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(ssao_prog)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, RT.gbuf_normal)
            glUniform1i(ssao_loc['gNormal'],0)
            glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, RT.gbuf_depth)
            glUniform1i(ssao_loc['gDepth'],1)
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, RT.noise_tex)
            glUniform1i(ssao_loc['noiseTex'],2)
            glActiveTexture(GL_TEXTURE0)
            glUniformMatrix4fv(ssao_loc['projection'],1,GL_TRUE,proj)
            glUniformMatrix4fv(ssao_loc['inverseProjection'],1,GL_TRUE,inv_proj)
            glUniform1f(ssao_loc['radius'], float(ssao_radius))
            glUniform1f(ssao_loc['bias'], float(ssao_bias))
            glUniform1f(ssao_loc['power'], float(ssao_power))
            glUniform2f(ssao_loc['aoSize'], float(aw), float(ah))
            glUniform3fv(ssao_loc['samples'], 16, kernel)
            glBindVertexArray(qvao); glDrawArrays(GL_TRIANGLES,0,6)

            # Blur
            glBindFramebuffer(GL_FRAMEBUFFER, RT.ao_blur_fbo)
            glViewport(0,0,aw,ah)
            glClearColor(1,1,1,1); glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(blur_prog)
            glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, RT.ao_tex)
            glUniform1i(blur_loc['aoTex'],0)
            glUniform2f(blur_loc['texelSize'], 1.0/aw, 1.0/ah)
            glDrawArrays(GL_TRIANGLES,0,6)
            glBindVertexArray(0)
            glEnable(GL_DEPTH_TEST)

        # Composite
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0,0,WIDTH,HEIGHT)
        glDisable(GL_DEPTH_TEST)
        glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(comp_prog)
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, RT.gbuf_color)
        glUniform1i(comp_loc['sceneColor'],0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, RT.ao_blur_tex if ssao_enabled else RT.ao_tex)
        glUniform1i(comp_loc['aoTex'],1)
        glUniform1i(comp_loc['useAO'], 1 if ssao_enabled else 0)
        glUniform1f(comp_loc['aoStrength'], float(ao_strength))
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(qvao); glDrawArrays(GL_TRIANGLES,0,6)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

        # ImGui
        imgui.new_frame()
        imgui.set_next_window_size((360, 700), imgui.Cond_.first_use_ever)
        imgui.begin("FPS Test - Kontroller")
        imgui.text("Fare Kilidi: " + ("ACIK (Q birak)" if mouse_locked else "SERBEST (Q kilitle)"))
        imgui.text("FPS: %.1f" % (1.0/dt if dt > 0 else 0.0))
        imgui.separator()
        imgui.text("Oyuncu")
        _, move_speed = imgui.slider_float("Hareket Hizi", move_speed, 1.0, 25.0, "%.2f")
        imgui.text("Konum: (%.2f, %.2f, %.2f)" % tuple(player_pos))
        imgui.separator()
        imgui.text("Yonlu Isik")
        ch, lc = imgui.color_edit3("Isik Rengi", light_color)
        if ch:
            light_color[0]=float(lc[0]); light_color[1]=float(lc[1]); light_color[2]=float(lc[2])
        _, light_intensity = imgui.slider_float("Parlaklik", light_intensity, 0.0, 5.0, "%.2f")
        _, light_azimuth = imgui.slider_float("Yatay Aci", light_azimuth, 0.0, 360.0, "%.1f")
        _, light_elevation = imgui.slider_float("Dikey Aci", light_elevation, 1.0, 89.0, "%.1f")
        ch, ac = imgui.color_edit3("Ortam Isigi", ambient_color)
        if ch:
            ambient_color[0]=float(ac[0]); ambient_color[1]=float(ac[1]); ambient_color[2]=float(ac[2])
        imgui.separator()
        imgui.text("Golgeler")
        _, shadows_enabled = imgui.checkbox("Golgeleri Ac", shadows_enabled)
        _, shadow_bias = imgui.slider_float("Shadow Bias", shadow_bias, 0.0001, 0.02, "%.5f")
        imgui.separator()
        imgui.text("SSAO (yari cozunurluk)")
        _, ssao_enabled = imgui.checkbox("SSAO Ac", ssao_enabled)
        _, ssao_radius = imgui.slider_float("SSAO Yaricap", ssao_radius, 0.05, 2.0, "%.3f")
        _, ssao_bias = imgui.slider_float("SSAO Bias", ssao_bias, 0.001, 0.15, "%.4f")
        _, ssao_power = imgui.slider_float("SSAO Guc", ssao_power, 0.3, 4.0, "%.2f")
        _, ao_strength = imgui.slider_float("AO Karisim", ao_strength, 0.0, 1.0, "%.2f")
        imgui.separator()
        imgui.text("Global Illumination (SH probe)")
        _, gi_enabled = imgui.checkbox("GI Ac", gi_enabled)
        _, gi_intensity = imgui.slider_float("GI Yogunluk", gi_intensity, 0.0, 4.0, "%.2f")
        _, gi_bounces = imgui.slider_int("Bounce", gi_bounces, 0, 4)
        imgui.text_wrapped("WASD | Fare | Q: fare kilidi | ESC: cikis")
        imgui.end()
        imgui.render(); impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown(); glfw.terminate()

if __name__ == "__main__":
    main()