import subprocess
import joblib
import numpy as np
import pandas as pd

# Chargement du modèle et du scaler
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# Fonction SNMP
def snmp_get(oid):
    cmd = ["snmpget", "-v2c", "-c", "public", "192.168.103.159", oid]
    try:
        output = subprocess.check_output(cmd, text=True)
        val = int(output.strip().split()[-1])
        return val
    except Exception as e:
        print(f"Erreur SNMP {oid}: {e}")
        return 0

# Fonction principale
def main():
    oids = [
        "1.3.6.1.2.1.2.2.1.13.11",  # ifInDiscards
        "1.3.6.1.2.1.2.2.1.19.11",  # ifOutDiscards
        "1.3.6.1.2.1.6.12.0",       # tcpRetransSegs
        "1.3.6.1.2.1.6.7.0",        # tcpOutRsts
        "1.3.6.1.2.1.6.14.0",       # tcpEstabResets
        "1.3.6.1.2.1.7.3.0",        # udpInErrors
        "1.3.6.1.2.1.7.4.0",        # udpNoPorts
        "1.3.6.1.2.1.4.10.0",       # ipOutDiscards
        "1.3.6.1.2.1.4.8.0",        # ipInDiscards
        "1.3.6.1.2.1.4.11.0",       # ipOutNoRoutes
        "1.3.6.1.2.1.4.3.0",        # ipInAddrErrors
        "1.3.6.1.2.1.5.3.0",        # icmpInDestUnreachs
        "1.3.6.1.2.1.5.15.0",       # icmpOutDestUnreachs
        "1.3.6.1.2.1.2.2.1.10.11",  # ifInOctets
        "1.3.6.1.2.1.2.2.1.16.11",  # ifOutOctets
        "1.3.6.1.2.1.6.11.0",       # tcpInSegs
        "1.3.6.1.2.1.6.10.0",       # tcpOutSegs
        "1.3.6.1.2.1.4.3.0",        # ipInReceives
        "1.3.6.1.2.1.4.10.0",       # ipOutRequests
    ]

    feature_names = [
        "ifInDiscards11", "ifoutDiscards11", "tcpRetransSegs", "tcpOutRsts",
        "tcpEstabResets", "udpInErrors", "udpNoPorts", "ipOutDiscards",
        "ipInDiscards", "ipOutNoRoutes", "ipInAddrErrors", "icmpInDestUnreachs",
        "icmpOutDestUnreachs", "ifInOctets11", "ifOutOctets11", "tcpInSegs",
        "tcpOutSegs", "ipInReceives", "ipOutRequests"
    ]

    # Récupération des valeurs SNMP
    features = [snmp_get(oid) for oid in oids]
    print("Données brutes:", features)

    # Création du DataFrame
    X_df = pd.DataFrame([features], columns=feature_names)

    # Normalisation
    X_scaled = scaler.transform(X_df)

    # Recréation du DataFrame normalisé avec noms de colonnes
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Prédiction
    classes = ['bruteForce', 'httpFlood', 'icmp-echo', 'normal', 'slowloris', 'slowpost', 'tcp-syn', 'udp-flood']
    pred = knn.predict(X_scaled_df)
    print("Prédiction (classe):", classes[pred[0]])

if __name__ == "__main__":
    main()