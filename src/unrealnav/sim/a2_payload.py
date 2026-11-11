"""The payload the A2 carries: a backpack frame with a Jetson AGX Thor on it, 4.9 kg.

Read from a URDF of the robot with its backpack (``a2_with_backpack_v7.urdf``, under
``external/a2_payload``; not included in this repository). Every part of it is fixed to the base without a
rotation, so each becomes boxes on the base body: its collision boxes collide, its visual
boxes are drawn, and its mass sits at its own origin as the URDF has it (the URDF gives the
parts no inertia of their own worth the name; a point mass each, 4.9 kg over a frame 35 cm
long, is what is taken).

**The gait was not trained with it.** ``scripts/check_a2_gait.py --payload`` says what it does
to the gait; ``a2_spec(payload=True)`` adds it.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import mujoco
import numpy as np

_HERE = Path(__file__).resolve().parents[3] / "external/a2_payload"
# The second of the user's two files (2026-10-10): the same parts and masses, with a streamlined cover as a mesh
# where the first had a box. The first is used where the second is not there.
# The newest first: the shell with its rear restored and an opening left for the A2's rear ports (2026-10-10, 18:02), the
# streamlined shell before it, the first frame without a shell. The three weigh the same and collide the same; they differ
# in what is drawn.
PAYLOAD_URDF = next((p for p in (_HERE / "thor_rear_shell/urdf/a2_with_backpack_v7.urdf", _HERE / "streamlined/urdf/a2_with_backpack_v7.urdf",
                                 _HERE / "a2_with_backpack_v7.urdf") if p.exists()),
                    _HERE / "a2_with_backpack_v7.urdf")
COVER = (0.84, 0.78, 0.66, 1.0)  # the cover: beige (the user's choice, 2026-10-10)
BLACK = (0.03, 0.03, 0.035, 1.0)  # the honeycomb adapter under it: black
PREFIX = "v7_"  # its links' names
DARK, CLEAR = (0.16, 0.17, 0.19, 1.0), (0.75, 0.82, 0.88, 0.35)  # anodised frame and electronics; the acrylic and the enclosure


def _xyz(element) -> list[float]:
    return [float(v) for v in element.get("xyz", "0 0 0").split()] if element is not None else [0.0, 0.0, 0.0]


def parts(urdf: Path = PAYLOAD_URDF) -> list[dict]:
    """Each part of the payload: its name, where its origin is in the base's frame, its mass, and its boxes
    (``kind``, ``size`` as full lengths, ``pos`` in the base's frame)."""
    if not urdf.exists():
        raise FileNotFoundError(f"{urdf}: the A2's payload is read from a URDF that is not included: put it under external/a2_payload")
    robot = ElementTree.parse(urdf).getroot()
    origin = {"base_link": [0.0, 0.0, 0.0]}
    joints = [j for j in robot.findall("joint") if j.get("type") == "fixed"]
    for _ in range(len(joints)):  # parents before children, whatever the file's order
        for joint in joints:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent in origin and child not in origin:
                turned = joint.find("origin").get("rpy", "0 0 0").split() if joint.find("origin") is not None else ["0"] * 3
                if child.startswith(PREFIX) and any(abs(float(v)) > 1e-9 for v in turned):
                    raise ValueError(f"{child} is turned against the base; this reads parts that are not")
                origin[child] = [a + b for a, b in zip(origin[parent], _xyz(joint.find("origin")))]
    found = []
    for link in robot.findall("link"):
        name = link.get("name")
        if not name.startswith(PREFIX):
            continue
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
        centre = [a + b for a, b in zip(origin[name], _xyz(inertial.find("origin") if inertial is not None else None))]
        boxes = []
        for kind in ("collision", "visual"):
            for shape in link.findall(kind):
                box, mesh = shape.find("geometry").find("box"), shape.find("geometry").find("mesh")
                place = [a + b for a, b in zip(origin[name], _xyz(shape.find("origin")))]
                if box is not None:
                    boxes.append({"kind": kind, "size": [float(v) for v in box.get("size").split()], "pos": place})
                elif mesh is not None and kind == "visual":
                    boxes.append({"kind": "mesh", "file": (urdf.parent / mesh.get("filename")).resolve(), "pos": place,
                                  "scale": [float(v) for v in mesh.get("scale", "1 1 1").split()]})
                else:
                    raise ValueError(f"{name} has a {kind} shape that is neither a box nor a mesh to draw")
        found.append({"name": name, "origin": origin[name], "centre": centre, "mass": mass, "boxes": boxes})
    return found


def in_metres(file: Path, scale: list[float]) -> Path:
    """A copy of a binary STL with its vertices scaled, kept beside it: the cover is drawn in millimetres, and what
    imports the robot file into Unreal is not trusted with a scale."""
    if all(abs(v - 1.0) < 1e-12 for v in scale):
        return file
    raw = file.read_bytes()
    # Named by what is in it: a new shell under the old file name must not be taken for the old one by whatever keeps
    # imported meshes by name (Unreal does).
    scaled = file.with_name(f"{file.stem}_{hashlib.sha256(raw).hexdigest()[:6]}_m.stl")
    if not scaled.exists():
        count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
        if len(raw) != 84 + 50 * count:
            raise ValueError(f"{file} is not a binary STL")
        triangles = np.frombuffer(raw, dtype=np.dtype([("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attribute", "<u2")]), count=count, offset=84).copy()
        triangles["points"] *= np.array(scale, dtype=np.float32)
        scaled.write_bytes(raw[:84] + triangles.tobytes())
    return scaled


def add_payload(spec: mujoco.MjSpec, urdf: Path = PAYLOAD_URDF) -> float:
    """Put the payload on the spec's base, as a body of its own fixed to it; returns its mass. Collision boxes go in
    the model's own collision class (and so collide and are not drawn), visual boxes are drawn and touch nothing.

    A body of its own because the base has its inertia written out, and MuJoCo then takes no mass from the base's
    shapes: boxes hung on the base itself weighed nothing (the first version of this: 40.07 kg with and without)."""
    found = parts(urdf)
    masses = np.array([part["mass"] for part in found])
    centres = np.array([part["centre"] for part in found])
    total = float(masses.sum())
    centre = (centres * masses[:, None]).sum(axis=0) / total
    arms = centres - centre
    inertia = sum(m * ((a @ a) * np.eye(3) - np.outer(a, a)) for m, a in zip(masses, arms))
    pack = spec.body("base_link").add_body(name="payload", mass=total, ipos=centre.tolist(), explicitinertial=True,
                                           fullinertia=[inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]])
    collision = next(g for g in spec.geoms if g.classname.name == "collision").classname
    covered = any(box["kind"] == "mesh" for part in found for box in part["boxes"])  # then what is under the cover is not drawn
    for part in found:
        clear = any(word in part["name"] for word in ("acrylic", "enclosure"))
        for k, box in enumerate(part["boxes"]):
            if box["kind"] == "mesh":
                file = in_metres(box["file"], box["scale"])
                spec.add_mesh(name=file.stem, file=str(file))
                pack.add_geom(name=f"{part['name']}_m{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=file.stem, pos=box["pos"],
                              contype=0, conaffinity=0, group=1, rgba=list(DARK if "vent" in file.stem else COVER))
                continue
            # Under a cover only what shows is drawn: the adapter below it, and the Thor, seen through the shell's rear opening.
            if box["kind"] == "visual" and covered and not any(word in part["name"] for word in ("honeycomb", "jetson")):
                continue
            half = [v / 2.0 for v in box["size"]]
            if box["kind"] == "collision":
                pack.add_geom(collision, name=f"{part['name']}_c{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=box["pos"])
            else:
                pack.add_geom(name=f"{part['name']}_v{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=box["pos"],
                              contype=0, conaffinity=0, group=1,
                              rgba=list(BLACK if "honeycomb" in part["name"] else CLEAR if clear else DARK))
    return total
