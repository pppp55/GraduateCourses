import pandas as pd
import numpy as np
from pathlib import Path
import json


class Vehicle:
    def __init__(self, v_id, specs):
        self.v_id = v_id
        self.model = specs["model"]
        self.capacity = specs["tank_capacity"]
        self.c_rate = specs["carbon_rate"]
        self.data = None


class STLOS_Engine:
    def __init__(self, data_path, cost_per_liter=7.5, tax_multiplier=0.1):
        self.root = Path(data_path)
        self.__cost_rate = cost_per_liter
        self.__tax_multiplier = tax_multiplier
        self.fleet = {}
        self._vehicle_lookup = {}
        self.master_telemetry = pd.DataFrame()

    @property
    def fuel_price(self):
        return self.__cost_rate

    @property
    def tax_rate(self):
        return self.__tax_multiplier

    """You need to implement '_parse_specs' and '_load_telemetry', which will be 
    used in 'ingest_fleet_data' """

    def _parse_specs(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            specs = json.load(f)
        return specs

    def _load_telemetry(self, file_path):
        """Robust loading with header sanitization."""
        df = pd.read_csv(file_path)
        df.columns = (
            pd.Index(df.columns)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        if "timestamp" not in df.columns:
            raise ValueError(f"Missing 'timestamp' column in {file_path}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def ingest_fleet_data(self):
        """Recursively crawl. Only targets telemetry.csv within vehicle folders."""
        for spec_path in self.root.rglob("specs.json"):
            district = spec_path.parent.parent.name
            v_id = spec_path.parent.name
            vehicle_key = f"{district}/{v_id}"
            specs = self._parse_specs(spec_path)
            v_obj = Vehicle(v_id, specs)
            t_path = spec_path.parent / "telemetry.csv"
            if t_path.exists():
                df = self._load_telemetry(t_path)
                df = self.clean_telemetry(df)
                v_obj.data = self.remove_outliers(df, v_obj.capacity)
                v_obj.data["district"] = district
                v_obj.data["vehicle_key"] = vehicle_key
                self.fleet[vehicle_key] = v_obj
                self._vehicle_lookup.setdefault(v_id, vehicle_key)

    def clean_telemetry(self, df):
        out = df.copy()
        out.columns = (
            pd.Index(out.columns)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        expected = ["timestamp", "lat", "lon", "fuel_level"]
        out = out[expected]
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
        out["fuel_level"] = pd.to_numeric(out["fuel_level"], errors="coerce")
        out = out.sort_values("timestamp").reset_index(drop=True)

        out[["lat", "lon"]] = out[["lat", "lon"]].interpolate(
            method="linear", limit_direction="both"
        )
        out = out.dropna(subset=["timestamp", "lat", "lon", "fuel_level"])
        return out

    def remove_outliers(self, df, cap):
        return df.loc[(df["fuel_level"] >= 0) & (df["fuel_level"] <= cap)].copy()

    def align_time_series(self):
        for vehicle in self.fleet.values():
            df = vehicle.data.copy()
            df = df.sort_values("timestamp").set_index("timestamp")
            district_val = (
                df["district"].iloc[0]
                if "district" in df.columns and not df.empty
                else np.nan
            )
            vehicle_key_val = (
                df["vehicle_key"].iloc[0]
                if "vehicle_key" in df.columns and not df.empty
                else np.nan
            )

            numeric = df.select_dtypes(include=[np.number])
            aligned = numeric.resample("5min").mean(numeric_only=True).ffill()
            aligned["district"] = district_val
            aligned["vehicle_key"] = vehicle_key_val

            vehicle.data = aligned.reset_index()

    """create 'self.master_telemetry' which will be used later"""

    def merge_districts(self):
        frames = []
        for key, vehicle in self.fleet.items():
            if vehicle.data is None or vehicle.data.empty:
                continue
            df = vehicle.data.copy()
            if "timestamp" not in df.columns and df.index.name == "timestamp":
                df = df.reset_index()
            df["v_id"] = vehicle.v_id
            df["fleet_key"] = key
            df["carbon_rate"] = vehicle.c_rate
            frames.append(df)

        if frames:
            self.master_telemetry = pd.concat(frames, ignore_index=True)
            self.master_telemetry["timestamp"] = pd.to_datetime(
                self.master_telemetry["timestamp"], errors="coerce"
            )
            self.master_telemetry = self.master_telemetry.sort_values(
                ["fleet_key", "timestamp"]
            ).reset_index(drop=True)
        else:
            self.master_telemetry = pd.DataFrame()

        return self.master_telemetry

    def compute_haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1 = np.asarray(lat1, dtype=float)
        lon1 = np.asarray(lon1, dtype=float)
        lat2 = np.asarray(lat2, dtype=float)
        lon2 = np.asarray(lon2, dtype=float)

        r = 6371.0
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        d_phi = np.radians(lat2 - lat1)
        d_lam = np.radians(lon2 - lon1)

        a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
            np.sin(d_lam / 2.0) ** 2
        )
        return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    def calculate_velocity_vector(self, v_id):
        key = self._vehicle_lookup.get(v_id, v_id)
        vehicle = self.fleet.get(key)
        if vehicle is None or vehicle.data is None or vehicle.data.empty:
            raise KeyError(f"Vehicle '{v_id}' not found in fleet.")

        df = vehicle.data.sort_values("timestamp")
        lat = df["lat"].to_numpy(dtype=float)
        lon = df["lon"].to_numpy(dtype=float)
        dist = self.compute_haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt_hours = 5.0 / 60.0
        return dist / dt_hours

    def detect_idling_events(self, eps=1e-5):
        if self.master_telemetry is None or self.master_telemetry.empty:
            self.merge_districts()

        df = self.master_telemetry.copy()
        df = df.sort_values(["fleet_key", "timestamp"]).reset_index(drop=True)
        g = df.groupby("fleet_key", sort=False)
        lat_delta = (df["lat"] - g["lat"].shift(1)).abs().fillna(np.inf)
        lon_delta = (df["lon"] - g["lon"].shift(1)).abs().fillna(np.inf)
        fuel_delta = (df["fuel_level"] - g["fuel_level"].shift(1)).fillna(0.0)
        df["is_idling"] = (lat_delta < eps) & (lon_delta < eps) & (fuel_delta < 0)

        self.master_telemetry = df
        return df.loc[df["is_idling"]].copy()

    def compute_fuel_burn_rate(self):
        if self.master_telemetry is None or self.master_telemetry.empty:
            self.merge_districts()

        df = self.master_telemetry.copy()
        df = df.sort_values(["fleet_key", "timestamp"]).reset_index(drop=True)
        g = df.groupby("fleet_key", sort=False)

        prev_lat = g["lat"].shift(1)
        prev_lon = g["lon"].shift(1)
        prev_fuel = g["fuel_level"].shift(1)
        same_vehicle = df["fleet_key"].eq(g["fleet_key"].shift(1))

        dist_delta = self.compute_haversine_distance(
            prev_lat.fillna(df["lat"]).to_numpy(),
            prev_lon.fillna(df["lon"]).to_numpy(),
            df["lat"].to_numpy(),
            df["lon"].to_numpy(),
        )
        dist_delta = np.where(same_vehicle.to_numpy(), dist_delta, 0.0)

        fuel_used = (prev_fuel - df["fuel_level"]).fillna(0.0)
        fuel_used = np.where(
            same_vehicle.to_numpy(), fuel_used.to_numpy(dtype=float), 0.0
        )
        burn_rate = np.divide(
            fuel_used,
            dist_delta,
            out=np.zeros_like(dist_delta, dtype=float),
            where=dist_delta > 0,
        )

        df["dist_delta"] = dist_delta
        df["fuel_delta"] = fuel_used
        df["burn_rate"] = burn_rate

        self.master_telemetry = df
        return df

    def find_nearest_refuel_station(self, station_csv):
        if self.master_telemetry is None or self.master_telemetry.empty:
            self.merge_districts()

        stations = pd.read_csv(station_csv)
        stations.columns = (
            pd.Index(stations.columns)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        s_lat = stations["lat"].to_numpy(dtype=float)
        s_lon = stations["lon"].to_numpy(dtype=float)
        t_lat = self.master_telemetry["lat"].to_numpy(dtype=float)
        t_lon = self.master_telemetry["lon"].to_numpy(dtype=float)

        dist_matrix = self.compute_haversine_distance(
            t_lat[:, None], t_lon[:, None], s_lat[None, :], s_lon[None, :]
        )
        nearest_idx = np.argmin(dist_matrix, axis=1)

        self.master_telemetry["nearest_station_idx"] = nearest_idx
        if "station_id" in stations.columns:
            station_ids = stations["station_id"].to_numpy()
            self.master_telemetry["nearest_station_id"] = station_ids[nearest_idx]

        return self.master_telemetry[
            ["fleet_key", "timestamp", "nearest_station_idx"]
        ].copy()

    def calculate_fleet_carbon_tax(self):
        if self.master_telemetry is None or self.master_telemetry.empty:
            self.merge_districts()
        if "dist_delta" not in self.master_telemetry.columns:
            self.compute_fuel_burn_rate()

        total_tax = float(
            np.sum(
                self.master_telemetry["dist_delta"].to_numpy(dtype=float)
                * self.master_telemetry["carbon_rate"].to_numpy(dtype=float)
            )
            * self.tax_rate
        )
        return total_tax

    def generate_district_report(self):
        if self.master_telemetry is None or self.master_telemetry.empty:
            self.merge_districts()

        report = (
            self.master_telemetry.groupby("district", as_index=True)["fuel_level"]
            .mean()
            .to_frame(name="mean_fuel_level")
        )
        return report

    def export_final_stats(self, path):
        if self.master_telemetry is None or self.master_telemetry.empty:
            self.merge_districts()

        if "is_idling" not in self.master_telemetry.columns:
            self.detect_idling_events()

        total_tax = self.calculate_fleet_carbon_tax()
        payload = {
            "total_tax": float(total_tax),
            "fleet_size": int(len(self.fleet)),
            "avg_fuel_level": float(self.master_telemetry["fuel_level"].mean()),
            "total_idling_events": int(self.master_telemetry["is_idling"].sum()),
        }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload
