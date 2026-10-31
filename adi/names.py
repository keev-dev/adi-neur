"""Names and colours for the 32 seats.

A name belongs to the seat, not to the body. Bodies die. The seat remains.
"""

NAMES = [
    "Brahma", "Vishnu", "Shiva", "Indra", "Agni", "Vayu", "Varuna", "Surya",
    "Chandra", "Yama", "Kubera", "Ganesha", "Kartikeya", "Hanuman", "Lakshmi", "Saraswati",
    "Parvati", "Durga", "Kali", "Rama", "Krishna", "Arjuna", "Bhima", "Yudhishthira",
    "Nakula", "Sahadeva", "Karna", "Drona", "Bhishma", "Ashwatthama", "Draupadi", "Sita",
]


def hue_for(slot: int, n: int = 32) -> float:
    return (slot * 360.0 / max(1, n) + 10.0) % 360.0


def vitality(energy: float, eaten: float, age: int, gen: int) -> float:
    """One number for colour intensity.

    It mixes three changes: current energy, recent foraging, and lineage depth.
    """
    e = max(0.0, min(1.0, energy / 120.0))
    rate = eaten / max(1.0, age / 300.0)
    f = max(0.0, min(1.0, rate / 8.0))
    g = max(0.0, min(1.0, gen / 40.0))
    return max(0.05, min(1.0, 0.50 * e + 0.30 * f + 0.20 * g))
