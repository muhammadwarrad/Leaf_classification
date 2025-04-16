import os
import shutil

source_dir = "C:/Users/moewa/Projects/ComputerVesion/flavia_raw/Leaves"
dest_dir = "C:/Users/moewa/Projects/ComputerVesion/leaf_dataset"

species_ranges = [
    ("Camptotheca_acuminata", 1001, 1059),
    ("Acer_palmatum", 1060, 1118),
    ("Acer_pseudoplatanus", 1119, 1177),
    ("Aesculus_hippocastanum", 1178, 1236),
    ("Betula_pendula", 1237, 1295),
    ("Broussonetia_papyrifera", 1296, 1354),
    ("Castanea_mollissima", 1355, 1413),
    ("Celtis_sinensis", 1414, 1472),
    ("Cercis_chinensis", 1473, 1531),
    ("Cornus_officinalis", 1532, 1590),
    ("Crateva_religiosa", 1591, 1649),
    ("Cudrania_tricuspidata", 1650, 1708),
    ("Diospyros_kaki", 1709, 1767),
    ("Ficus_microcarpa", 1768, 1826),
    ("Ginkgo_biloba", 1827, 1885),
    ("Ilex_cornuta", 1886, 1944),
    ("Koelreuteria_paniculata", 1945, 2003),
    ("Lagerstroemia_indica", 2004, 2062),
    ("Liquidambar_formosana", 2063, 2121),
    ("Liriodendron_chinense", 2122, 2180),
    ("Magnolia_grandiflora", 2181, 2239),
    ("Magnolia_liliflora", 2240, 2298),
    ("Malus_pumila", 2299, 2357),
    ("Melia_azedarach", 2358, 2416),
    ("Morus_alba", 2417, 2475),
    ("Phellodendron_amurense", 2476, 2534),
    ("Populus_tomentosa", 2535, 2593),
    ("Prunus_persica", 2594, 2652),
    ("Prunus_salicina", 2653, 2711),
    ("Quercus_acutissima", 2712, 2770),
    ("Rhus_chinensis", 2771, 2829),
    ("Salix_babylonica", 2830, 2907)
]

os.makedirs(dest_dir, exist_ok=True)

for species_name, start, end in species_ranges:
    species_dir = os.path.join(dest_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)
    for img_id in range(start, end + 1):
        img_name = f"{img_id:04d}.jpg"  
        src_path = os.path.join(source_dir, img_name)
        dest_path = os.path.join(species_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")
        else:
            print(f"Image {img_name} not found")

print("Dataset organization complete!")