"""Predict a house price with the trained model.

Usage:
    python predict.py
    python predict.py --medinc 5.2 --houseage 20 --averooms 6 --avebedrms 1.1 \
        --population 1500 --aveoccup 3 --latitude 34.05 --longitude -118.25
"""
import argparse

import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Predict California house price")
    p.add_argument("--medinc", type=float, default=3.8, help="Median income (tens of thousands $)")
    p.add_argument("--houseage", type=float, default=28)
    p.add_argument("--averooms", type=float, default=5.4)
    p.add_argument("--avebedrms", type=float, default=1.1)
    p.add_argument("--population", type=float, default=1400)
    p.add_argument("--aveoccup", type=float, default=3.0)
    p.add_argument("--latitude", type=float, default=34.05)
    p.add_argument("--longitude", type=float, default=-118.25)
    a = p.parse_args()

    bundle = joblib.load("models/best_model.joblib")
    row = {
        "MedInc": a.medinc, "HouseAge": a.houseage, "AveRooms": a.averooms,
        "AveBedrms": a.avebedrms, "Population": a.population, "AveOccup": a.aveoccup,
        "Latitude": a.latitude, "Longitude": a.longitude,
    }
    row["RoomsPerPerson"] = row["AveRooms"] / max(row["AveOccup"], 1e-3)
    row["BedroomRatio"] = row["AveBedrms"] / max(row["AveRooms"], 1e-3)
    X = pd.DataFrame([row])[bundle["features"]]
    price = bundle["model"].predict(X)[0] * 100_000
    print(f"Predicted house price: ${price:,.0f}")


if __name__ == "__main__":
    main()
