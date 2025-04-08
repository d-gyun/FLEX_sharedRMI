import numpy as np
import osmium
import os

class NodeCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.coords = []

    def node(self, n):
        if n.location.valid():
            self.coords.append((n.location.lon, n.location.lat))

def extract_coordinates_from_pbf(pbf_file, size=1000000, seed=42):
    handler = NodeCollector()
    handler.apply_file(pbf_file, locations=True)
    coords = handler.coords

    print(f"[INFO] Total nodes extracted: {len(coords)}")
    if len(coords) < size:
        raise ValueError("Not enough nodes in the PBF file.")

    np.random.seed(seed)
    sampled = np.array(coords)
    indices = np.random.choice(len(sampled), size=size, replace=False)
    sampled = sampled[indices]
    return sampled

class DataGen:
    def __init__(self, size, range_size, pbf_file=None):
        self.size = size
        self.range_size = range_size
        self.pbf_file = pbf_file

    def rand_data(self):
        data = [np.random.randint(self.range_size) for _ in range(self.size)]
        return np.array(data)

    def log_normal(self):
        mu, sigma = 0, 1
        data = np.random.lognormal(mean=mu, sigma=sigma, size=self.size * 10)
        data = np.unique(data)
        while len(data) < self.size:
            additional_data = np.random.lognormal(mean=mu, sigma=sigma, size=self.size)
            data = np.unique(np.concatenate((data, additional_data)))
        data = np.sort(data[:self.size])
        data = data / np.max(data) * self.range_size
        return np.rint(data)

    def generate_longitudes(self):
        coords = extract_coordinates_from_pbf(self.pbf_file, self.size)
        longitudes = coords[:, 0]
        return np.sort(longitudes)

    def generate_longlat(self):
        coords = extract_coordinates_from_pbf(self.pbf_file, self.size)
        longitudes = coords[:, 0]
        latitudes = coords[:, 1]
        longlat_values = 180 * np.floor(longitudes) + latitudes
        return np.sort(longlat_values)


# 데이터셋 생성 및 저장 함수
def generate_and_save_data():
    size = 1000000
    range_size = 10000000
    output_dir = "../datasets"
    os.makedirs(output_dir, exist_ok=True)

    pbf_file_path = "../osm/california-latest.osm.pbf"
    gen = DataGen(size=size, range_size=range_size, pbf_file=pbf_file_path)
    generators = {
        # "RANDOM": gen.rand_data,
        # "LOGNORMAL": gen.log_normal,
        "LONGITUDES": gen.generate_longitudes,
        "LONGLAT": gen.generate_longlat
    }

    for name, gen_func in generators.items():
        data = gen_func()
        file_path = f"{output_dir}/dataset_{name}_{size}.csv"
        np.savetxt(file_path, data, delimiter=",")
        print(f"Saved: {file_path}")

    return

if __name__ == '__main__':
    generate_and_save_data()
