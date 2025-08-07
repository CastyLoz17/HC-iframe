import turtle
from math import *
import time
from typing import Union
import keyboard
import mouse

try:
    using_mouse = True
    import mouse
except (ImportError, OSError):
    using_mouse = False
    from pynput.mouse import Listener as MouseListener


Number = Union[int, float]


class Vector2:
    def __init__(self, x: Number, y: Number):
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Number) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Number) -> "Vector2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector2":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar: Number) -> "Vector2":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2(self.x // scalar, self.y // scalar)

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: "Vector2") -> bool:
        # Use small epsilon for floating point comparison
        epsilon = 1e-10
        return abs(self.x - other.x) < epsilon and abs(self.y - other.y) < epsilon

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        """More efficient than magnitude() when you only need to compare magnitudes"""
        return self.x * self.x + self.y * self.y

    def normalize(self) -> "Vector2":
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)

    def distance_to(self, other: "Vector2") -> float:
        return (self - other).magnitude()

    def angle(self) -> float:
        """Returns the angle of the vector in radians"""
        return atan2(self.y, self.x)

    def rotate(self, angle: float) -> "Vector2":
        """Rotate the vector by the given angle in radians"""
        cos_a, sin_a = cos(angle), sin(angle)
        return Vector2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    def is_zero(self) -> bool:
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon


class Vector3:
    def __init__(self, x: Number, y: Number, z: Number):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Number) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Number) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __floordiv__(self, scalar: Number) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x // scalar, self.y // scalar, self.z // scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: "Vector3") -> bool:
        epsilon = 1e-10
        return (
            abs(self.x - other.x) < epsilon
            and abs(self.y - other.y) < epsilon
            and abs(self.z - other.z) < epsilon
        )

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10), round(self.z, 10)))

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalize(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()

    def is_zero(self) -> bool:
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon and abs(self.z) < epsilon

    def project_onto(self, other: "Vector3") -> "Vector3":
        if other.is_zero():
            return Vector3(0, 0, 0)
        return other * (self.dot(other) / other.magnitude_squared())

    def reflect(self, normal: "Vector3") -> "Vector3":
        return self - 2 * self.project_onto(normal)

    def to_radians(self):
        return Vector3(*[radians(i) for i in self])


class LightingConfig:
    """Configuration class for lighting settings"""

    def __init__(
        self,
        base_brightness=0.3,  # Minimum brightness level (0.0 to 1.0)
        max_brightness=1.0,  # Maximum brightness level (0.0 to 1.0)
        falloff_rate=0.05,  # How fast brightness decreases with distance
        falloff_type="linear",  # "linear", "quadratic", "exponential", or "none"
        distance_scale=1.0,
    ):  # Scale factor for distance calculations
        self.base_brightness = max(0.0, min(1.0, base_brightness))
        self.max_brightness = max(0.0, min(1.0, max_brightness))
        self.falloff_rate = max(0.0, falloff_rate)
        self.falloff_type = falloff_type
        self.distance_scale = max(0.01, distance_scale)

    def calculate_brightness(self, distance):
        """Calculate brightness based on distance and lighting configuration"""
        scaled_distance = distance * self.distance_scale

        if self.falloff_type == "none":
            return self.max_brightness
        elif self.falloff_type == "linear":
            brightness = self.max_brightness - (scaled_distance * self.falloff_rate)
        elif self.falloff_type == "quadratic":
            brightness = self.max_brightness - (
                scaled_distance * scaled_distance * self.falloff_rate
            )
        elif self.falloff_type == "exponential":
            brightness = self.max_brightness * exp(-scaled_distance * self.falloff_rate)
        else:
            brightness = self.max_brightness - (scaled_distance * self.falloff_rate)

        return max(self.base_brightness, min(self.max_brightness, brightness))


def zero2() -> Vector2:
    return Vector2(0, 0)


def zero3() -> Vector3:
    return Vector3(0, 0, 0)


def unit_x2() -> Vector2:
    return Vector2(1, 0)


def unit_y2() -> Vector2:
    return Vector2(0, 1)


def unit_x3() -> Vector3:
    return Vector3(1, 0, 0)


def unit_y3() -> Vector3:
    return Vector3(0, 1, 0)


def unit_z3() -> Vector3:
    return Vector3(0, 0, 1)


class Camera:
    def __init__(self, pos, pitch, yaw, lighting_config=None):
        self.pos = pos
        self.pitch = pitch
        self.yaw = yaw
        self.lighting_config = lighting_config or LightingConfig()

        turtle.tracer(0)
        turtle.setup(800, 600)
        pen = turtle.Turtle()
        pen.hideturtle()
        pen.penup()
        pen.color("black")
        pen.speed(0)
        self.pen = pen

    def set_lighting_config(self, lighting_config):
        self.lighting_config = lighting_config

    def move_axis(self, pos):
        self.pos += pos
        return self.pos

    def move(self, steps, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(-sin(self.yaw), 0, cos(self.yaw))
        else:
            forward = self.get_view_direction()

        move = forward * steps
        self.pos += move
        return self.pos

    def strafe(self, steps):
        right = Vector3(cos(self.yaw), 0, sin(self.yaw))

        move = right * steps
        self.pos += move
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        self.move(pos.z, horizontal_only)
        self.strafe(pos.x)
        self.pos.y -= pos.y

    def rotate(self, pitch, yaw):
        self.pitch += pitch
        self.yaw += yaw

        max_pitch = radians(89)
        self.pitch = max(-max_pitch, min(max_pitch, self.pitch))

    def get_view_direction(self):
        return Vector3(
            sin(self.yaw) * cos(self.pitch),
            -sin(self.pitch),
            -cos(self.yaw) * cos(self.pitch),
        )

    def project_point(self, point, screen_width=800, screen_height=600, fov=90):
        relative = point - self.pos

        yaw = self.yaw
        pitch = self.pitch

        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        cos_pitch = cos(pitch)
        sin_pitch = sin(pitch)

        xz = Vector3(
            cos_yaw * relative.x + sin_yaw * relative.z,
            relative.y,
            -sin_yaw * relative.x + cos_yaw * relative.z,
        )

        final = Vector3(
            xz.x,
            cos_pitch * xz.y - sin_pitch * xz.z,
            sin_pitch * xz.y + cos_pitch * xz.z,
        )

        if final.z <= 0:
            return None

        aspect_ratio = screen_width / screen_height

        fov_rad = radians(fov)
        f = 1 / tan(fov_rad / 2)

        screen_x = (final.x * f / final.z) * (screen_height / screen_width)
        screen_y = final.y * f / final.z

        screen_x = screen_x * (screen_width / 2) + screen_width / 2
        screen_y = screen_y * (screen_height / 2) + screen_height / 2

        turtle_x = screen_x - screen_width / 2
        turtle_y = screen_y - screen_height / 2

        return (turtle_x, turtle_y)

    def compute_face_normal(self, face):
        if len(face) < 3:
            return Vector3(0, 0, 1)

        edge1 = face[1] - face[0]
        edge2 = face[2] - face[0]
        normal = Vector3(
            edge1.y * edge2.z - edge1.z * edge2.y,
            edge1.z * edge2.x - edge1.x * edge2.z,
            edge1.x * edge2.y - edge1.y * edge2.x,
        ).normalize()

        return normal

    def is_face_visible(self, face, normal):
        return True
        # if len(face) < 3:
        #     return True

        # face_center = face[0]
        # view_vector = (face_center - self.pos).normalize()
        # dot_product = normal.dot(view_vector)

        # return True
        # abc123
        # return dot_product < 1

    def render(self, objects, materials, screen_width=800, screen_height=600, fov=90):
        rendered_faces = []
        for object in objects:
            for face, material in object["faces"]:
                normal = self.compute_face_normal(face)

                if not self.is_face_visible(face, normal):
                    continue

                projected = [
                    self.project_point(point, screen_width, screen_height, fov)
                    for point in face
                ]

                if not projected or None in projected:
                    continue

                centroid = sum(face, Vector3(0, 0, 0)) / len(face)
                distance = (centroid - self.pos).magnitude()

                brightness = self.lighting_config.calculate_brightness(distance)

                if material in materials:
                    base_color = materials[material]
                    color = [max(0, min(1, v * brightness)) for v in base_color]
                else:
                    color = [brightness, brightness, brightness]

                rendered_faces.append((projected, color, distance))

        rendered_faces.sort(key=lambda x: x[2], reverse=True)

        for projected, color, distance in rendered_faces:
            self.pen.goto(*projected[0])
            self.pen.fillcolor(color)
            self.pen.begin_fill()

            for pos in projected[1:]:
                self.pen.goto(*pos)

            self.pen.end_fill()


def rotate_point_around_axis(point, anchor, axis, angle):
    p = point - anchor
    k = axis.normalize()

    cos_a = cos(angle)
    sin_a = sin(angle)

    rotated = (
        p * cos_a
        + Vector3(k.y * p.z - k.z * p.y, k.z * p.x - k.x * p.z, k.x * p.y - k.y * p.x)
        * sin_a
        + k * (k.dot(p)) * (1 - cos_a)
    )

    return rotated + anchor


def rotate_objects(objects, anchor, rotation_vector):
    angle = sqrt(rotation_vector.x**2 + rotation_vector.y**2 + rotation_vector.z**2)

    if angle == 0:
        return

    axis = Vector3(
        rotation_vector.x / angle, rotation_vector.y / angle, rotation_vector.z / angle
    )

    for obj in objects:
        for face_idx, (face, material) in enumerate(obj["faces"]):
            rotated_face = []
            for vertex in face:
                rotated_vertex = rotate_point_around_axis(vertex, anchor, axis, angle)
                rotated_face.append(rotated_vertex)
            obj["faces"][face_idx] = (rotated_face, material)


def move_objects(objects, axis):
    for obj in objects:
        for face_idx, (face, material) in enumerate(obj["faces"]):
            for vertex in face:
                for i in range(len(face)):
                    face[i] = face[i] + axis


def load_obj(file_path, scale=0):
    with open(file_path) as file:
        lines = file.readlines()

    objects = []
    vertexes = []
    curr_mtl = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        command, *args = parts

        if command == "o":
            name = args[0]
            objects.append({"name": name, "faces": []})
        elif command == "v":
            vertexes.append(Vector3(*map(lambda x: float(x) * scale, args)))
        elif command == "f":

            objects[-1]["faces"].append(
                ([vertexes[int(i.split("/")[0]) - 1] for i in args], curr_mtl)
            )
        elif command == "usemtl":
            curr_mtl = args[0]

    return objects


def load_mtl(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    colors = {}
    curr_mtl = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        command, *args = parts
        if command == "newmtl":
            curr_mtl = args[0]
        elif command == "Kd" and curr_mtl:
            colors[curr_mtl] = [float(i) for i in args]
    return colors


def mouse_init():
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_listener
    mouse_delta_x = 0
    mouse_delta_y = 0
    last_mouse_x = 0
    last_mouse_y = 0
    mouse_initialized = False
    if not using_mouse:
        mouse_listener = MouseListener(on_move=on_mouse_move)


def on_mouse_move(x, y):
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_locked

    if not mouse_locked:
        return

    if not mouse_initialized:
        last_mouse_x = x
        last_mouse_y = y
        mouse_initialized = True
        if not using_mouse:
            mouse_listener.start()
        else:
            mouse.move(400, 300)
        return

    mouse_delta_x = x - last_mouse_x
    mouse_delta_y = y - last_mouse_y

    last_mouse_x = x
    last_mouse_y = y


def handle_movement(speed=0.2, sensitivity=0.05):
    global mouse_delta_x, mouse_delta_y, using_mouse

    camera_movement = zero3()
    camera_angle = zero2()

    if keyboard.is_pressed("w"):
        camera_movement.z = speed
    if keyboard.is_pressed("s"):
        camera_movement.z = -speed
    if keyboard.is_pressed("a"):
        camera_movement.x = -speed
    if keyboard.is_pressed("d"):
        camera_movement.x = speed
    if keyboard.is_pressed("ctrl"):
        camera_movement.y = speed
    if keyboard.is_pressed("space"):
        camera_movement.y = -speed

    if mouse_locked:
        if using_mouse:
            try:
                x, y = mouse.get_position()
                if mouse_initialized:
                    camera_angle.x -= (y - last_mouse_y) * sensitivity
                    camera_angle.y -= (x - last_mouse_x) * sensitivity
                    mouse.move(400, 300)
                else:
                    globals()["mouse_initialized"] = True
                globals()["last_mouse_x"] = 400
                globals()["last_mouse_y"] = 300
            except Exception:
                camera_angle.x += mouse_delta_y * sensitivity
                camera_angle.y -= mouse_delta_x * sensitivity
                mouse_delta_x = 0
                mouse_delta_y = 0
        else:
            camera_angle.x += mouse_delta_y * sensitivity
            camera_angle.y -= mouse_delta_x * sensitivity
            mouse_delta_x = 0
            mouse_delta_y = 0

    return camera_movement, camera_angle


def normalize_framerate(target):
    def decorator(func):
        def wrapped(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            time_to_sleep = max(0, (1 / (target + 2)) - elapsed)
            time.sleep(time_to_sleep)
            return result

        return wrapped

    return decorator


if __name__ == "__main__":
    turtle.bgcolor("#000000")

    lighting = LightingConfig(
        base_brightness=0.2,
        max_brightness=1.0,
        falloff_rate=0.07,
        falloff_type="linear",  # linear, quadratic, exponential, none
        distance_scale=1.0,
    )

    camera = Camera(Vector3(1, 0, -7), radians(0), radians(0), lighting)
    cube = load_obj("objs/blahaj.obj", scale=2.5)
    material = load_mtl("objs/blahaj.mtl")

    mouse_locked = True
    momentum = zero3()

    camera.pen.clear()
    rotate_objects(cube, zero3(), Vector3(radians(16), radians(-45), radians(10)))
    camera.render(cube, material, fov=90)
    turtle.update()
    turtle.done()
