import numpy as np


def update_chain(chain, delta_angles):
    chain = np.array(chain, np.float32)
    delta_angles = np.array(delta_angles, np.float32)

    for end_molecule in range(len(chain), 3, -1):
        forward_change(chain[end_molecule-4:end_molecule])

    chain[3:, 2] += delta_angles

    for end_molecule in range(4, len(chain) + 1):
        inverse_change(chain[end_molecule-4:end_molecule])

    return chain


def forward_change(sub_chain):
    print("Start", sub_chain)

    translate(sub_chain)
    print("Translated", sub_chain)

    rotate(sub_chain)
    print("Rotated", sub_chain)

    resolve(sub_chain)
    print("Resolved", sub_chain)


def inverse_change(sub_chain):
    inv_resolve(sub_chain)
    print("Inv Resolved", sub_chain)

    inv_rotate(sub_chain)
    print("Inv Rotated", sub_chain)

    inv_translate(sub_chain)
    print("Inv Translated", sub_chain)


def translate(sub_chain):
    sub_chain[3] -= sub_chain[2]


def cross_product(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]], np.float32)


def get_basis(sub_chain):
    e1 = sub_chain[2] - sub_chain[1]
    e1 /= np.linalg.norm(e1)

    e2 = sub_chain[0] - sub_chain[1]
    e2 /= np.linalg.norm(e2)

    ez = e1
    ey = np.linalg.cross(e1, e2)
    ey /= np.linalg.norm(ey)
    ex = np.linalg.cross(ey, ez)

    print("Basis", ex, ey, ez)

    return ex, ey, ez


def get_rotation_matrix(sub_chain):
    basis = get_basis(sub_chain)

    matrix = np.zeros((3, 3), np.float32)
    matrix[0] = basis[0]
    matrix[1] = basis[1]
    matrix[2] = basis[2]

    return matrix


def rotate(sub_chain):
    sub_chain[3] = get_rotation_matrix(sub_chain) @ sub_chain[3]


def resolve(sub_chain):
    r4 = sub_chain[3]

    r = np.linalg.norm(r4)
    theta = np.arccos(r4[2] / r)
    phi = np.arccos(r4[0] / np.linalg.norm(r4[:2]))

    if r4[1] < 0:
        phi = 2 * np.pi - phi

    sub_chain[3] = (r, theta, phi)


def inv_translate(sub_chain):
    sub_chain[3] += sub_chain[2]


def inv_rotate(sub_chain):
    matrix = get_rotation_matrix(sub_chain)
    sub_chain[3] = matrix.T @ sub_chain[3]


def inv_resolve(sub_chain):
    r, theta, phi = sub_chain[3]
    sub_chain[3] = (
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    )


def translate_f():
    # Translate positions and apply identity to forces.
    pass


def rotate_f():
    # Rotate positions and forces.
    pass


result = update_chain([[-1, np.sqrt(3) / 2, 0], [-0.5, 0, 0], [0.5, 0, 0], [1, np.sqrt(3) / 2, 0]], [np.pi / 3])
print()
print("Result:", result)
