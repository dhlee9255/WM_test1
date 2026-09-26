"""Download Kenney packs (CC0) and patch character GLBs so panda3d-gltf loads them as animated Actors."""
import io
import json
import shutil
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "assets"
PACKS = {
    "mini-dungeon": "https://kenney.nl/media/pages/assets/mini-dungeon/6cd72dc849-1785314274/kenney_mini-dungeon.zip",
    "graveyard-kit": "https://kenney.nl/media/pages/assets/graveyard-kit/ba8d4b4517-1760691807/kenney_graveyard-kit_5.0.zip",
}
DUNGEON_GLB = ROOT / "mini-dungeon" / "Models" / "GLB format"
GRAVEYARD_GLB = ROOT / "graveyard-kit" / "Models" / "GLB format"
PROCESSED = ROOT / "processed"
GRAVEYARD_CHARS = ["character-skeleton", "character-zombie"]


def read_glb(path):
    data = path.read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])
    rest = data[20 + json_len:]
    bin_len = struct.unpack("<I", rest[:4])[0] if rest else 0
    return gltf, bytearray(rest[8:8 + bin_len])


def write_glb(path, gltf, binary):
    binary = bytes(binary) + b"\0" * ((4 - len(binary) % 4) % 4)
    if gltf.get("buffers"):
        gltf["buffers"][0]["byteLength"] = len(binary)
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    body = struct.pack("<I", len(js)) + b"JSON" + js + struct.pack("<I", len(binary)) + b"BIN\0" + binary
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)


def merge_skins(src: Path, dst: Path):
    # Mini Dungeon characters bind body and head to two identical skins; panda3d-gltf
    # crashes on that, so point both meshes at the first skin.
    gltf, binary = read_glb(src)
    for node in gltf["nodes"]:
        if "skin" in node:
            node["skin"] = 0
    gltf["skins"] = gltf["skins"][:1]
    write_glb(dst, gltf, binary)


def _trs(node):
    t = np.array(node.get("translation", [0, 0, 0]), np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), np.float64)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * s
    m[:3, 3] = t
    return m


def rigid_to_skinned(src: Path, dst: Path):
    # Graveyard characters animate body-part nodes directly (no skin). Turn every part node
    # into a joint, bake each part mesh into model space and bind it 100% to its joint.
    gltf, binary = read_glb(src)
    nodes, acc, views = gltf["nodes"], gltf["accessors"], gltf["bufferViews"]
    root = gltf["scenes"][0]["nodes"][0]
    stack, joints, world = [(c, np.eye(4)) for c in nodes[root]["children"]], [], {}
    while stack:
        n, parent = stack.pop(0)
        joints.append(n)
        world[n] = parent @ _trs(nodes[n])
        stack.extend((c, world[n]) for c in nodes[n].get("children", []))

    def transform_vec3(idx, mat, is_point, width=3):
        a, v = acc[idx], views[acc[idx]["bufferView"]]
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride", width * 4)
        for i in range(a["count"]):
            o = off + i * stride
            vec = np.array(struct.unpack_from("<3f", binary, o))
            vec = mat[:3, :3] @ vec + (mat[:3, 3] if is_point else 0)
            if not is_point:
                vec /= max(np.linalg.norm(vec), 1e-9)
            struct.pack_into("<3f", binary, o, *vec)
        if is_point and "min" in a:
            pts = [struct.unpack_from("<3f", binary, off + i * stride) for i in range(a["count"])]
            a["min"], a["max"] = list(np.min(pts, 0)), list(np.max(pts, 0))

    def add_accessor(raw, comp_type, count, typ):
        nonlocal binary
        binary += b"\0" * ((4 - len(binary) % 4) % 4)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(raw)})
        binary += raw
        gltf["accessors"].append({"bufferView": len(gltf["bufferViews"]) - 1, "componentType": comp_type,
                                  "count": count, "type": typ})
        return len(gltf["accessors"]) - 1

    ibm_raw = b"".join(struct.pack("<16f", *np.linalg.inv(world[n]).T.flatten()) for n in joints)
    ibm = add_accessor(ibm_raw, 5126, len(joints), "MAT4")
    gltf["skins"] = [{"joints": joints, "inverseBindMatrices": ibm}]
    for ji, n in enumerate(joints):
        mesh = nodes[n].pop("mesh", None)
        if mesh is None:
            continue
        for prim in gltf["meshes"][mesh]["primitives"]:
            attrs = prim["attributes"]
            transform_vec3(attrs["POSITION"], world[n], True)
            if "NORMAL" in attrs:
                transform_vec3(attrs["NORMAL"], world[n], False)
            if "TANGENT" in attrs:
                transform_vec3(attrs["TANGENT"], world[n], False, width=4)
            count = gltf["accessors"][prim["attributes"]["POSITION"]]["count"]
            prim["attributes"]["JOINTS_0"] = add_accessor(bytes([ji, 0, 0, 0]) * count, 5121, count, "VEC4")
            prim["attributes"]["WEIGHTS_0"] = add_accessor(struct.pack("<4f", 1, 0, 0, 0) * count, 5126, count, "VEC4")
        nodes.append({"name": f"{nodes[n].get('name', n)}-mesh", "mesh": mesh, "skin": 0})
        nodes[root]["children"].append(len(nodes) - 1)
    write_glb(dst, gltf, binary)


def main():
    for name, url in PACKS.items():
        if (ROOT / name).exists():
            continue
        print("downloading", name)
        with urllib.request.urlopen(url) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(ROOT / name)
    (PROCESSED / "Textures").mkdir(parents=True, exist_ok=True)
    shutil.copy(DUNGEON_GLB / "Textures" / "colormap.png", PROCESSED / "Textures" / "colormap.png")
    for f in DUNGEON_GLB.glob("*.glb"):
        dst = PROCESSED / f.name
        if f.name.startswith("character-"):
            merge_skins(f, dst)
        else:
            shutil.copy(f, dst)
    # graveyard models use their own colormap, so keep them in a subfolder
    gy = PROCESSED / "graveyard"
    (gy / "Textures").mkdir(parents=True, exist_ok=True)
    shutil.copy(GRAVEYARD_GLB / "Textures" / "colormap.png", gy / "Textures" / "colormap.png")
    for f in GRAVEYARD_GLB.glob("*.glb"):
        if f.stem in GRAVEYARD_CHARS:
            rigid_to_skinned(f, gy / f.name)
        elif not f.stem.startswith("character-"):
            shutil.copy(f, gy / f.name)  # props: fences, gravestones, roads, trees...
    print("assets ready:", PROCESSED)


if __name__ == "__main__":
    main()
