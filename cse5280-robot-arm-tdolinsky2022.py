import pygame
from pygame.locals import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
from dataclasses import dataclass, field
from collections import deque
import math
import random

shader_program = None


# ============================================================
# SHADER / CAMERA / BASIC DRAW HELPERS
# ============================================================

def shader():
    global shader_program

    vertex_shader = """
    #version 120

    varying vec3 v_normal;
    varying vec3 v_position;
    varying vec4 v_color;

    void main()
    {
        v_normal = normalize(gl_NormalMatrix * gl_Normal);
        vec4 pos = gl_ModelViewMatrix * gl_Vertex;
        v_position = pos.xyz;
        v_color = gl_Color;

        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    }
    """

    fragment_shader = """
    #version 120

    varying vec3 v_normal;
    varying vec3 v_position;
    varying vec4 v_color;

    void main()
    {
        vec3 N = normalize(v_normal);
        vec3 V = normalize(-v_position);

        vec3 light_dir = normalize(vec3(-0.55, 1.0, 0.35));
        float diff = max(dot(N, light_dir), 0.0);

        float upness = N.y * 0.5 + 0.5;
        vec3 sky_color = vec3(1.05, 1.05, 1.08);
        vec3 ground_color = vec3(0.60, 0.60, 0.63);
        vec3 hemi = mix(ground_color, sky_color, upness);

        float ambient = 0.80;

        float floor_mask = smoothstep(0.88, 0.98, N.y);
        float height_mask = 1.0 - smoothstep(22.0, 28.0, v_position.y);
        float glossy_mask = floor_mask * height_mask;

        float ao = 1.0;
        ao -= (1.0 - max(N.y, 0.0)) * 0.22;

        float vertical_grad = clamp((v_position.y + 20.0) / 40.0, 0.7, 1.2);

        vec3 base = v_color.rgb;

        float gloss_strength = mix(0.08, 1.0, glossy_mask);
        float gloss_shininess = mix(20.0, 200.0, glossy_mask);

        vec3 R = reflect(-light_dir, N);
        float spec = pow(max(dot(V, R), 0.0), gloss_shininess) * gloss_strength;

        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.25 * glossy_mask;

        vec3 color = base * hemi * (ambient + 1.10 * diff) * ao * vertical_grad
                   + vec3(spec)
                   + vec3(fresnel);

        gl_FragColor = vec4(color, 1.0);
    }
    """

    shader_program = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )


def camera(x, y, z):
    glRotatef(x, 1, 0, 0)
    glRotatef(y, 0, 1, 0)
    glRotatef(z, 0, 0, 1)


def Grid():
    glColor3f(0.82, 0.82, 0.82)
    y = -0.8

    glBegin(GL_LINES)
    for i in range(51):
        glVertex3f(i, y, 0)
        glVertex3f(i, y, 50)

        glVertex3f(0, y, i)
        glVertex3f(50, y, i)
    glEnd()


# ============================================================
#   ROBOT PARTS
# ============================================================

# ============================================================
# BASIC SHAPES FOR ROBOT
# ============================================================

def draw_cylinder(radius, height, slices=24):
    q = gluNewQuadric()
    gluCylinder(q, radius, radius, height, slices, 1)
    gluDeleteQuadric(q)

def draw_sphere(radius, slices=20, stacks=20):
    q = gluNewQuadric()
    gluSphere(q, radius, slices, stacks)
    gluDeleteQuadric(q)

def draw_box(sx, sy, sz, color):
    glPushMatrix()
    glColor3f(*color)
    glScalef(sx, sy, sz)
    Block(-0.5, -0.5, -0.5, 1.0, color)
    glPopMatrix()

# ============================================================
# DRAW ROBOT 
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def solve_robot_ik(robot_x, robot_y, robot_z, target_x, target_z):
    """
    2-link IK that prefers an inward/downward elbow pose.
    The arm is modeled along local +Y, so we convert the target into
    the robot's local vertical plane and solve for an elbow-down pose.
    """

    # shoulder pivot height in world space
    shoulder_y = robot_y + 6.0

    # red dot is on ground floor
    target_y = 2.5

    dx = target_x - robot_x
    dz = target_z - robot_z

    # rotate robot so it faces target in XZ plane
    base_yaw = math.degrees(math.atan2(dz, dx))

    # horizontal distance from shoulder to target
    horizontal = math.hypot(dx, dz)

    # convert target into 2D arm plane
    px = horizontal
    py = target_y - shoulder_y

    L1 = 10.0
    L2 = 10.0

    dist = math.hypot(px, py)
    dist = clamp(dist, 0.001, L1 + L2 - 0.001)

    # recompute clamped point along same direction
    if math.hypot(px, py) > 1e-8:
        scale = dist / math.hypot(px, py)
        px *= scale
        py *= scale

    # elbow-down solution
    cos_elbow = clamp((px*px + py*py - L1*L1 - L2*L2) / (2.0 * L1 * L2), -1.0, 1.0)
    elbow_rad = -math.acos(cos_elbow)

    k1 = L1 + L2 * math.cos(elbow_rad)
    k2 = L2 * math.sin(elbow_rad)
    shoulder_rad = math.atan2(py, px) - math.atan2(k2, k1)

    shoulder_deg = math.degrees(shoulder_rad)
    elbow_deg = math.degrees(elbow_rad)

    return base_yaw, shoulder_deg, elbow_deg


def draw_robot(x, y, z, target_x, target_z):
    dark = (0.15, 0.16, 0.18)
    mid = (0.45, 0.48, 0.55)
    light = (0.85, 0.87, 0.90)
    accent = (0.10, 0.85, 0.95)
    gold = (0.80, 0.65, 0.20)

    base_yaw, shoulder_angle, elbow_angle = solve_robot_ik(x, y, z, target_x, target_z)

    glPushMatrix()
    glTranslatef(x, y, z)

    # face toward red dot
    glRotatef(-base_yaw, 0, 1, 0)

    # base
    draw_box(4.5, 2.0, 4.5, dark)

    glPushMatrix()
    glTranslatef(0, 3.5, 0)
    draw_box(2.2, 7.0, 2.2, mid)
    glPopMatrix()

    # shoulder pivot
    glTranslatef(0, 6.5, 0)
    glColor3f(*gold)
    draw_sphere(1.0)

    # first arm segment
    glPushMatrix()
    glRotatef(shoulder_angle - 90.0, 0, 0, 1)

    glPushMatrix()
    glTranslatef(0, 5.0, 0)
    draw_box(1.6, 10.0, 1.6, mid)
    glPopMatrix()

    # elbow pivot
    glTranslatef(0, 10.0, 0)
    glColor3f(*light)
    draw_sphere(0.9)

    # second arm segment
    glRotatef(elbow_angle, 0, 0, 1)

    glPushMatrix()
    glTranslatef(0, 5.0, 0)
    draw_box(1.3, 10.0, 1.3, light)
    glPopMatrix()

    # wrist
    glTranslatef(0, 10.0, 0)
    glColor3f(*accent)
    draw_sphere(0.5)

    # pen points a bit downward
    glRotatef(-65, 0, 0, 1)

    glColor3f(0.1, 0.2, 0.8)
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    draw_cylinder(0.15, 2.5)
    glPopMatrix()

    glColor3f(0.95, 0.95, 0.95)
    glPushMatrix()
    glTranslatef(0, -2.5, 0)
    glRotatef(-90, 1, 0, 0)
    q = gluNewQuadric()
    gluCylinder(q, 0.15, 0.02, 0.5, 16, 1)
    gluDeleteQuadric(q)
    glPopMatrix()

    glPopMatrix()
    glPopMatrix()

    
# ============================================================
# ROBOT FUNCTIONS
# ============================================================
def get_agents_near_exits(agents):
    near_exit = []
    for a in agents:
        if a["done"]:
            continue

        if a["floor"] != 0:
            continue

        x, z = a["pos"][0], a["pos"][2]

        # near back exit
        if 7 <= x <= 13 and z <= 3:
            near_exit.append(a)

        # near left exit
        elif x <= 3 and 5 <= z <= 12:
            near_exit.append(a)

    return near_exit

def compute_cluster_center(agent_list):
    if len(agent_list) == 0:
        return None

    avg_x = sum(a["pos"][0] for a in agent_list) / len(agent_list)
    avg_z = sum(a["pos"][2] for a in agent_list) / len(agent_list)

    return (avg_x, avg_z)

def predict_cluster_position(agent_list, dt=2.0):
    if len(agent_list) == 0:
        return None

    avg_x = sum(a["pos"][0] for a in agent_list) / len(agent_list)
    avg_z = sum(a["pos"][2] for a in agent_list) / len(agent_list)

    avg_vx = sum(a["vel"][0] for a in agent_list) / len(agent_list)
    avg_vz = sum(a["vel"][1] for a in agent_list) / len(agent_list)

    future_x = avg_x + avg_vx * dt
    future_z = avg_z + avg_vz * dt

    return (future_x, future_z)


# ============================================================
#   BUILDING
# ============================================================

def Quad(v1, v2, v3, v4, color, normal):
    glColor3f(*color)
    glBegin(GL_QUADS)
    glNormal3f(*normal)
    glVertex3f(*v1)
    glVertex3f(*v2)
    glVertex3f(*v3)
    glVertex3f(*v4)
    glEnd()


def Block(x, y, z, size,
          top_color, side_color=None, bottom_color=None,
          top=True, bottom=True,
          left=True, right=True,
          front=True, back=True):

    if side_color is None:
        side_color = top_color
    if bottom_color is None:
        bottom_color = side_color

    x2 = x + size
    y2 = y + size
    z2 = z + size

    if top:
        Quad((x, y2, z), (x2, y2, z), (x2, y2, z2), (x, y2, z2), top_color, (0, 1, 0))
    if bottom:
        Quad((x, y, z), (x2, y, z), (x2, y, z2), (x, y, z2), bottom_color, (0, -1, 0))
    if left:
        Quad((x, y, z), (x, y, z2), (x, y2, z2), (x, y2, z), side_color, (-1, 0, 0))
    if right:
        Quad((x2, y, z), (x2, y, z2), (x2, y2, z2), (x2, y2, z), side_color, (1, 0, 0))
    if front:
        Quad((x, y, z2), (x2, y, z2), (x2, y2, z2), (x, y2, z2), side_color, (0, 0, 1))
    if back:
        Quad((x, y, z), (x2, y, z), (x2, y2, z), (x, y2, z), side_color, (0, 0, -1))


def FloorSlab(x1, y, z1, x2, z2, thickness, floor_color, edge_color):
    y_top = y
    y_bot = y - thickness

    Quad((x1, y_top, z1), (x2, y_top, z1), (x2, y_top, z2), (x1, y_top, z2),
         floor_color, (0, 1, 0))
    Quad((x1, y_bot, z1), (x2, y_bot, z1), (x2, y_bot, z2), (x1, y_bot, z2),
         edge_color, (0, -1, 0))

    Quad((x1, y_bot, z1), (x1, y_bot, z2), (x1, y_top, z2), (x1, y_top, z1),
         edge_color, (-1, 0, 0))
    Quad((x2, y_bot, z1), (x2, y_bot, z2), (x2, y_top, z2), (x2, y_top, z1),
         edge_color, (1, 0, 0))
    Quad((x1, y_bot, z1), (x2, y_bot, z1), (x2, y_top, z1), (x1, y_top, z1),
         edge_color, (0, 0, -1))
    Quad((x1, y_bot, z2), (x2, y_bot, z2), (x2, y_top, z2), (x1, y_top, z2),
         edge_color, (0, 0, 1))


def ramp(x1, y1, z1, x2, y2, z2, thickness, top_color, bottom_color):
    glColor3f(*top_color)
    glBegin(GL_TRIANGLES)
    glNormal3f(0, 1, 0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y1, z2)

    glNormal3f(0, 1, 0)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y2, z2)
    glVertex3f(x2, y2, z1)
    glEnd()

    y1_bot = y1 - thickness
    y2_bot = y2 - thickness

    glColor3f(*bottom_color)
    glBegin(GL_TRIANGLES)
    glNormal3f(0, -1, 0)
    glVertex3f(x1, y1_bot, z1)
    glVertex3f(x1, y1_bot, z2)
    glVertex3f(x2, y2_bot, z2)

    glNormal3f(0, -1, 0)
    glVertex3f(x1, y1_bot, z1)
    glVertex3f(x2, y2_bot, z2)
    glVertex3f(x2, y2_bot, z1)
    glEnd()

    glColor3f(*bottom_color)
    glBegin(GL_QUADS)
    glNormal3f(-1, 0, 0)
    glVertex3f(x1, y1_bot, z1)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x2, y2_bot, z1)
    glEnd()

    glBegin(GL_QUADS)
    glNormal3f(1, 0, 0)
    glVertex3f(x1, y1_bot, z2)
    glVertex3f(x2, y2_bot, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x1, y1, z2)
    glEnd()

    glBegin(GL_QUADS)
    glNormal3f(0, 0, -1)
    glVertex3f(x1, y1_bot, z1)
    glVertex3f(x2, y2_bot, z1)
    glVertex3f(x2, y2, z1)
    glVertex3f(x1, y1, z1)
    glEnd()

    glBegin(GL_QUADS)
    glNormal3f(0, 0, 1)
    glVertex3f(x1, y1_bot, z2)
    glVertex3f(x1, y1, z2)
    glVertex3f(x2, y2, z2)
    glVertex3f(x2, y2_bot, z2)
    glEnd()


def short_wall(x, y, z, length, height, color, orientation='x'):
    for i in range(length):
        for j in range(height):
            if orientation == 'x':
                Block(x + i, y + j, z, 1.0, color)
            else:
                Block(x, y + j, z + i, 1.0, color)


# ============================================================
# BUILDING DRAWING
# ============================================================

def Building():
    wall_top = (0.50, 0.50, 0.53)
    wall_side = (0.40, 0.40, 0.43)
    wall_bottom = (0.34, 0.34, 0.36)

    floor_top = (0.66, 0.66, 0.70)
    floor_side = (0.46, 0.46, 0.50)

    block = 1.0
    width = 20
    height = 30
    floor_levels = [0, 10, 20]
    floor_thickness = 1

    exit_x = 8
    exit_width = 4
    exit_height = 7

    # back wall z = 0
    for x in range(width):
        for y in range(height):
            if exit_x <= x < exit_x + exit_width and y < exit_height:
                continue

            Block(
                x, y, 0, block,
                wall_top, wall_side, wall_bottom,
                top=(y == height - 1),
                bottom=(y == 0),
                left=(x == 0),
                right=(x == width - 1),
                front=True,
                back=True
            )

        # left wall x = 0
        for z in range(width):
            for y in range(height):
                if 6 <= z <= 10 and 0 <= y <= 5:
                    continue

                Block(
                    0, y, z, block,
                    wall_top, wall_side, wall_bottom,
                    top=(y == height - 1),
                    bottom=(y == 0),
                    left=True,
                    right=True,
                    front=(z == width - 1),
                    back=(z == 0)
                )

    short_wall(5, 10, 10, 10, 2, wall_side, orientation='x')
    short_wall(8, 10, 10, 10, 2, wall_side, orientation='z')
    short_wall(8, 20, 10, 10, 2, wall_side, orientation='z')

    for level in floor_levels:
        if level == 10:
            hole_x1, hole_x2, hole_z1, hole_z2 = 9, 16, 13, 18
        elif level == 20:
            hole_x1, hole_x2, hole_z1, hole_z2 = 3, 15, 5, 10
        else:
            FloorSlab(0, level, 0, width, width, floor_thickness, floor_top, floor_side)
            continue

        if hole_x1 > 0:
            FloorSlab(0, level, 0, hole_x1, width, floor_thickness, floor_top, floor_side)
        if hole_x2 < width:
            FloorSlab(hole_x2, level, 0, width, width, floor_thickness, floor_top, floor_side)
        if hole_z1 > 0:
            FloorSlab(hole_x1, level, 0, hole_x2, hole_z1, floor_thickness, floor_top, floor_side)
        if hole_z2 < width:
            FloorSlab(hole_x1, level, hole_z2, hole_x2, width, floor_thickness, floor_top, floor_side)

    ramp_thickness = 0.8
    ramp_top = (0.8, 0.2, 0.9)
    ramp_bottom = (0.6, 0.1, 0.7)

    ramp(5, 0, 13, 15, 8, 18, ramp_thickness, ramp_top, ramp_bottom)
    ramp(15, 10, 6.5, 5, 18, 8.5, ramp_thickness, ramp_top, ramp_bottom)

# ============================================================
# NAV WORLD
# ============================================================

@dataclass
class FloorMap:
    width: int
    depth: int
    walkable: np.ndarray
    wall_dist: np.ndarray
    base_cost: np.ndarray
    goal_cells: list = field(default_factory=list)
    back_exit_cells: list = field(default_factory=list)
    left_exit_cells: list = field(default_factory=list)
    back_exit_cost: np.ndarray = None
    left_exit_cost: np.ndarray = None


@dataclass
class RampConnector:
    from_floor: int
    to_floor: int
    entry_rect: tuple          # x1, x2, z1, z2
    path_start: tuple          # (x, y, z) high/start point
    path_end: tuple            # (x, y, z) low/end point


@dataclass
class NavWorld:
    floors: dict
    ramps: list
    floor_y: dict


FLOOR_Y = {
    0: 2.5,
    1: 12.5,
    2: 22.5,
}

DIRS_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]


def in_bounds(x, z, w, d):
    return 0 <= x < w and 0 <= z < d


def cell_center(x, z):
    return (x + 0.5, z + 0.5)


def world_to_cell(x, z, width, depth):
    cx = int(np.clip(math.floor(x), 0, width - 1))
    cz = int(np.clip(math.floor(z), 0, depth - 1))
    return cx, cz


def bfs_distance(walkable, goals):
    w, d = walkable.shape
    dist = np.full((w, d), np.inf, dtype=np.float32)
    q = deque()

    for gx, gz in goals:
        if in_bounds(gx, gz, w, d) and walkable[gx, gz]:
            dist[gx, gz] = 0.0
            q.append((gx, gz))

    while q:
        x, z = q.popleft()
        for dx, dz in DIRS_8:
            nx, nz = x + dx, z + dz
            if not in_bounds(nx, nz, w, d):
                continue
            if not walkable[nx, nz]:
                continue

            step_cost = 1.4142 if dx != 0 and dz != 0 else 1.0
            nd = dist[x, z] + step_cost
            if nd < dist[nx, nz]:
                dist[nx, nz] = nd
                q.append((nx, nz))

    return dist


def build_nav_world():
    width = 20
    depth = 20

    floors = {}

    for floor_id in (0, 1, 2):
        walkable = np.ones((width, depth), dtype=bool)

        # outer boundaries
        walkable[0, :] = False
        walkable[width - 1, :] = False
        walkable[:, depth - 1] = False

        # back wall z=0 and left wall x=0
        walkable[:, 0] = False
        walkable[0, :] = False

        if floor_id == 0:
            # back exit: x = 8..11 at z = 0
            for x in range(8, 12):
                walkable[x, 0] = True

            # left exit: x = 0 at z = 6..10
            for z in range(6, 11):
                walkable[0, z] = True
        else:
            walkable[:, 0] = False

        floors[floor_id] = FloorMap(
            width=width,
            depth=depth,
            walkable=walkable,
            wall_dist=None,
            base_cost=None,
            goal_cells=[]
        )

    # short walls on 2nd floor
    for x in range(5, 15):
        floors[1].walkable[x, 10] = False
    for z in range(10, 20):
        floors[1].walkable[8, z] = False

    # short wall on 3rd floor
    for z in range(10, 20):
        floors[2].walkable[8, z] = False

    #go around ramp on 2nd floor
    for x in range(5, 15):
        for z in range(7, 18):
            floors[1].walkable[x, z] = False


    ramps = [
        # 3rd floor down to 2nd floor
        RampConnector(
            from_floor=2,
            to_floor=1,
            entry_rect=(4, 7, 8, 10),
            path_start=(5.5, FLOOR_Y[2], 9.5),     # top of upper ramp
            path_end=(14.5, FLOOR_Y[1], 5.5)       # bottom of upper ramp
        ),

        # 2nd floor down to 1st floor
        RampConnector(
            from_floor=1,
            to_floor=0,
            entry_rect = (9, 16, 14, 16),
            path_start=(14.5, FLOOR_Y[1], 17.5),   # top of lower ramp
            path_end=(5.5, FLOOR_Y[0], 13.5)       # bottom of lower ramp
        )
    ]

    # goals:
    floors[0].back_exit_cells = [(x, 0) for x in range(8, 12)]
    floors[0].left_exit_cells = [(0, z) for z in range(6, 11)]
    floors[0].goal_cells = floors[0].back_exit_cells + floors[0].left_exit_cells

    # floor 1 goal = first ramp entry zone
    floor1_goals = []
    x1, x2, z1, z2 = ramps[1].entry_rect
    for x in range(x1, x2):
        for z in range(z1, z2):
            if floors[1].walkable[x, z]:
                floor1_goals.append((x, z))
    floors[1].goal_cells = floor1_goals

    # floor 2 goal = second ramp entry zone
    floor2_goals = []
    x1, x2, z1, z2 = ramps[0].entry_rect
    for x in range(x1, x2):
        for z in range(z1, z2):
            if floors[2].walkable[x, z]:
                floor2_goals.append((x, z))
    floors[2].goal_cells = floor2_goals

    for floor_id, floor in floors.items():
        blocked = ~floor.walkable
        blocked_cells = [(x, z) for x in range(width) for z in range(depth) if blocked[x, z]]

        # distance to nearest blocked cell
        wall_dist = np.full((width, depth), 999.0, dtype=np.float32)
        for x in range(width):
            for z in range(depth):
                if not floor.walkable[x, z]:
                    wall_dist[x, z] = 0.0
                else:
                    best = 999.0
                    for bx, bz in blocked_cells:
                        dd = math.hypot(x - bx, z - bz)
                        if dd < best:
                            best = dd
                    wall_dist[x, z] = best

        dist_field = bfs_distance(floor.walkable, floor.goal_cells)

        base_cost = np.array(dist_field, copy=True)
        for x in range(width):
            for z in range(depth):
                if not floor.walkable[x, z]:
                    base_cost[x, z] = np.inf

        floor.wall_dist = wall_dist
        floor.base_cost = base_cost

        # extra cost maps for floor 0 so agents can commit to one exit
        if floor_id == 0:
            back_dist = bfs_distance(floor.walkable, floor.back_exit_cells)
            left_dist = bfs_distance(floor.walkable, floor.left_exit_cells)

            floor.back_exit_cost = np.array(back_dist, copy=True)
            floor.left_exit_cost = np.array(left_dist, copy=True)

            for x in range(width):
                for z in range(depth):
                    if not floor.walkable[x, z]:
                        floor.back_exit_cost[x, z] = np.inf
                        floor.left_exit_cost[x, z] = np.inf

    return NavWorld(
        floors=floors,
        ramps=ramps,
        floor_y=FLOOR_Y
    )

# ============================================================
# AGENTS
# ============================================================

def spawn_agent(x, z, floor, color=None, speed=None):
    if color is None:
        color = (
            random.uniform(0.1, 0.95),
            random.uniform(0.1, 0.95),
            random.uniform(0.1, 0.95)
        )

    if speed is None:
        speed = random.uniform(16.0, 22.0)

    if random.random() < 0.5:
        target_exit = "back"
    else:
        target_exit = "left"

    return {
        "pos": [x, FLOOR_Y[floor], z],
        "vel": [0.0, 0.0],
        "speed": speed,
        "radius": 0.35,
        "color": color,
        "floor": floor,
        "done": False,
        "on_ramp": False,
        "ramp": None,
        "target_exit": target_exit,
    }

def create_agents(world, total_agents=20):
    agents = []

    # Split agents across floors as evenly as possible
    floor_counts = {
        0: total_agents // 3,
        1: total_agents // 3,
        2: total_agents - 2 * (total_agents // 3)
    }

    for floor_id, count in floor_counts.items():
        floor = world.floors[floor_id]

        # good spawn cells = walkable and not too close to exits/walls
        valid_cells = []
        for x in range(2, floor.width - 2):
            for z in range(2, floor.depth - 2):
                if not floor.walkable[x, z]:
                    continue
                if floor.wall_dist[x, z] < 1.5:
                    continue
                valid_cells.append((x, z))

        random.shuffle(valid_cells)

        for i in range(min(count, len(valid_cells))):
            cx, cz = valid_cells[i]

            # small random offset inside the cell so they don't all sit dead-center
            px = cx + random.uniform(0.2, 0.8)
            pz = cz + random.uniform(0.2, 0.8)

            agents.append(spawn_agent(px, pz, floor_id))

    return agents



def crowd_penalty(agent, floor_id, nx, nz, agents):
    penalty = 0.0
    px, pz = cell_center(nx, nz)

    for other in agents:
        if other is agent or other["done"] or other["floor"] != floor_id:
            continue

        ox, oz = other["pos"][0], other["pos"][2]
        d = math.hypot(px - ox, pz - oz)
        if d < 2.0:
            penalty += (2.0 - d) * 2.4

    return penalty


def inertia_penalty(agent, move_dx, move_dz):
    vx, vz = agent["vel"]
    speed = math.hypot(vx, vz)
    if speed < 1e-6:
        return 0.0

    desired_len = math.hypot(move_dx, move_dz)
    if desired_len < 1e-6:
        return 0.0

    dot = (vx * move_dx + vz * move_dz) / (speed * desired_len)
    return (1.0 - dot) * 0.5


def evaluate_candidate(agent, world, agents, floor_id, nx, nz, move_dx, move_dz):
    floor = world.floors[floor_id]

    if not in_bounds(nx, nz, floor.width, floor.depth):
        return np.inf
    if not floor.walkable[nx, nz]:
        return np.inf

    if floor_id == 0:
        if agent["target_exit"] == "back":
            cost = floor.back_exit_cost[nx, nz]
        else:
            cost = floor.left_exit_cost[nx, nz]
    else:
        cost = floor.base_cost[nx, nz]

    # small LOCAL wall penalty only
    d = floor.wall_dist[nx, nz]
    if d < 1.35:
        cost += (1.35 - d) * 1.2


    # Robot obstacle penalty (ground floor only)
    if floor_id == 0:
        rx, rz = robot_effector
        px, pz = cell_center(nx, nz)
        d_robot = math.hypot(px - rx, pz - rz)

        if d_robot < 2.0:
            cost += (2.0 - d_robot) * 5.0

    # crowd spacing
    cost += crowd_penalty(agent, floor_id, nx, nz, agents)

    # small inertia penalty so motion stays smooth
    cost += inertia_penalty(agent, move_dx, move_dz)

    # tiny tie-break noise
    cost += random.uniform(0.0, 0.01)

    return cost


def maybe_start_ramp(agent, world):
    if agent["on_ramp"]:
        return

    current_floor = agent["floor"]
    x, z = agent["pos"][0], agent["pos"][2]

    for ramp_conn in world.ramps:
        if ramp_conn.from_floor != current_floor:
            continue

        x1, x2, z1, z2 = ramp_conn.entry_rect
        if x1 <= x < x2 and z1 <= z < z2:
            agent["on_ramp"] = True
            agent["ramp"] = ramp_conn
            return


def update_agent_on_ramp(agent):
    ramp_conn = agent["ramp"]
    if ramp_conn is None:
        agent["on_ramp"] = False
        return

    sx, sy, sz = ramp_conn.path_start
    ex, ey, ez = ramp_conn.path_end

    dx = ex - agent["pos"][0]
    dz = ez - agent["pos"][2]
    dist = math.hypot(dx, dz)

    if dist < 0.15:
        agent["pos"][0] = ex
        agent["pos"][1] = ey
        agent["pos"][2] = ez
        agent["floor"] = ramp_conn.to_floor
        agent["on_ramp"] = False
        agent["ramp"] = None
        agent["vel"][0] = 0.0
        agent["vel"][1] = 0.0
        return

    step = min(agent["speed"] * 0.12, dist)
    dir_x = dx / dist
    dir_z = dz / dist

    agent["pos"][0] += dir_x * step
    agent["pos"][2] += dir_z * step

    ramp_dx = ex - sx
    ramp_dz = ez - sz
    ramp_len_sq = ramp_dx * ramp_dx + ramp_dz * ramp_dz

    if ramp_len_sq > 1e-8:
        proj = ((agent["pos"][0] - sx) * ramp_dx + (agent["pos"][2] - sz) * ramp_dz) / ramp_len_sq
        proj = max(0.0, min(1.0, proj))
    else:
        proj = 1.0

    agent["pos"][1] = sy + (ey - sy) * proj
    agent["vel"][0] = dir_x * step
    agent["vel"][1] = dir_z * step


def snap_agent_to_floor_if_needed(agent, world):
    if not agent["on_ramp"]:
        agent["pos"][1] = world.floor_y[agent["floor"]]


def update_agents(agents, world):
    for agent in agents:
        if agent["done"]:
            continue

        # if already on ramp, only do ramp motion
        if agent["on_ramp"]:
            update_agent_on_ramp(agent)
            continue

        floor_id = agent["floor"]
        floor = world.floors[floor_id]

        cx, cz = world_to_cell(agent["pos"][0], agent["pos"][2], floor.width, floor.depth)

        # first-floor exit reached (either exit)
        if floor_id == 0 and (
            (8 <= cx <= 11 and cz == 0) or
            (cx == 0 and 6 <= cz <= 10)
        ):
            agent["done"] = True
            continue
        best_cost = np.inf
        best_move = (0, 0)
        best_cell = (cx, cz)

        stay_cost = evaluate_candidate(agent, world, agents, floor_id, cx, cz, 0, 0)
        best_cost = stay_cost

        for dx, dz in DIRS_8:
            nx, nz = cx + dx, cz + dz
            cost = evaluate_candidate(agent, world, agents, floor_id, nx, nz, dx, dz)
            if cost < best_cost:
                best_cost = cost
                best_move = (dx, dz)
                best_cell = (nx, nz)

        tx, tz = cell_center(best_cell[0], best_cell[1])

        move_x = tx - agent["pos"][0]
        move_z = tz - agent["pos"][2]
        move_len = math.hypot(move_x, move_z)

        if move_len > 1e-6:
            step = min(agent["speed"] * 0.02, move_len)
            dir_x = move_x / move_len
            dir_z = move_z / move_len

            agent["pos"][0] += dir_x * step
            agent["pos"][2] += dir_z * step
            agent["vel"][0] = dir_x * step
            agent["vel"][1] = dir_z * step
        else:
            agent["vel"][0] *= 0.5
            agent["vel"][1] *= 0.5

        maybe_start_ramp(agent, world)
        snap_agent_to_floor_if_needed(agent, world)


def draw_agents(agents):
    for a in agents:
        if a["done"]:
            continue

        glColor3f(*a["color"])
        glPushMatrix()
        glTranslatef(a["pos"][0], a["pos"][1], a["pos"][2])
        glScalef(1.0, 5.0, 2.0)
        quadric = gluNewQuadric()
        gluSphere(quadric, a["radius"], 16, 16)
        gluDeleteQuadric(quadric)
        glPopMatrix()


# ============================================================
# OPTIONAL DEBUG DRAWING FOR NAV CELLS
# ============================================================

def draw_debug_goals(world):
    glUseProgram(0)
    for floor_id, floor in world.floors.items():
        y = world.floor_y[floor_id] - 1.9
        for gx, gz in floor.goal_cells:
            glColor3f(1.0, 1.0, 0.0)
            glPushMatrix()
            glTranslatef(gx + 0.5, y, gz + 0.5)
            glut_like_cube(0.18)
            glPopMatrix()


def glut_like_cube(size):
    s = size
    Block(-s, -s, -s, 2 * s, (1.0, 1.0, 0.0))

def draw_effector(x, z):
    glColor3f(1.0, 0.0, 0.0)
    glPushMatrix()
    glTranslatef(x, 2.5, z)
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.5, 16, 16)
    gluDeleteQuadric(quadric)
    glPopMatrix()


# ============================================================
# MAIN
# ============================================================

robot_target = [10.0, 10.0]   # where robot pen tip should move
robot_effector = [10.0, 10.0] # actual current position

def main():
    pygame.init()
    display = (1000, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    shader()

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.72, 0.69, 0.62, 1.0)

    gluPerspective(45, (display[0] / display[1]), 0.1, 300.0)

    glTranslatef(-25, -10, -80)
    camera(10, -30, 0)
    glTranslatef(10, -5, -20)

    clock = pygame.time.Clock()


    world = build_nav_world()

    agents = create_agents(world, total_agents=20)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        SIM_STEPS = 1   # increase for smoother motion (try 5–10)

        for _ in range(SIM_STEPS):
            update_agents(agents, world)

            near_exit_agents = get_agents_near_exits(agents)
            cluster_center = compute_cluster_center(near_exit_agents)
            predicted = predict_cluster_position(near_exit_agents)

            if predicted is not None:
                dx = predicted[0] - robot_effector[0]
                dz = predicted[1] - robot_effector[1]
                dist = math.hypot(dx, dz)

                if dist > 0.1:
                    step_size = 0.08
                    robot_effector[0] += dx * step_size
                    robot_effector[1] += dz * step_size

                    # keep robot inside valid ground-floor area
                    robot_effector[0] = max(1.5, min(18.5, robot_effector[0]))
                    robot_effector[1] = max(1.5, min(18.5, robot_effector[1]))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(0)
        Grid()

        glUseProgram(shader_program)
        Building()
        draw_agents(agents)
        draw_robot(25, 0, 10, robot_effector[0], robot_effector[1])
        draw_effector(robot_effector[0], robot_effector[1])

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
