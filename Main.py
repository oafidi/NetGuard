import subprocess
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import time
import smtplib
from email.message import EmailMessage
from collections import deque
import csv
from datetime import datetime

# Configuration
SNMP_COMMUNITY = "secret"
SNMP_HOST = "192.168.103.81"
MODEL_FILE = "isolation_forest_model.pkl"
CSV_FILE = "snmp_monitoring_data.csv"
STATS_FILE = "snmp_monitoring_stats.csv"
TRAINING_SAMPLES = 50
SAMPLE_INTERVAL = 5
RETRAIN_INTERVAL = 300  # Réentraîner toutes les 5 minutes
EXPORT_INTERVAL = 60   # Exporter vers CSV toutes les minutes

OIDS = {
    "cpu_load": "1.3.6.1.4.1.2021.10.1.3.1",
    "ram_total": "1.3.6.1.4.1.2021.4.5.0",
    "ram_free": "1.3.6.1.4.1.2021.4.6.0",
    "processes": "1.3.6.1.4.1.2021.11.9.0",
    "ifInDiscards": "1.3.6.1.2.1.2.2.1.13.1",
    "ifOutDiscards": "1.3.6.1.2.1.2.2.1.19.1",
    "tcpRetransSegs": "1.3.6.1.2.1.6.12.0",
    "tcpOutRsts": "1.3.6.1.2.1.6.7.0",
    "sysUpTime": "1.3.6.1.2.1.1.3.0",
    "tcpActiveOpens": "1.3.6.1.2.1.6.5.0",
    "tcpPassiveOpens": "1.3.6.1.2.1.6.6.0",
    "udpInDatagrams": "1.3.6.1.2.1.7.1.0",
    "udpOutDatagrams": "1.3.6.1.2.1.7.4.0",
    "icmpInErrors": "1.3.6.1.2.1.5.14.0",
    "icmpOutErrors": "1.3.6.1.2.1.5.15.0",
}

EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "omarafidi2005@gmail.com"
SMTP_PASSWORD = "uvha wayi txgi dobi"
ALERT_EMAIL = "elbahchoumichaimae1@gmail.com"

# Buffer circulaire pour les données d'entraînement et réentraînement périodique
data_buffer = deque(maxlen=1000)
# Liste pour stocker toutes les données collectées avec timestamps et statuts
collected_data = []

def snmp_get(oid):
    start_time = time.time()
    cmd = ["snmpget", "-v2c", "-c", SNMP_COMMUNITY, SNMP_HOST, oid]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=5, stderr=subprocess.PIPE)
        response_time = time.time() - start_time
        parts = output.strip().split("=", 1)
        if len(parts) < 2:
            raise ValueError(f"Format SNMP inattendu pour OID {oid}: {output}")
        val_str = parts[1].strip()
        if val_str.startswith("Timeticks:"):
            val = float(val_str.split("(")[1].split(")")[0])
        elif val_str.startswith("INTEGER:"):
            val = float(val_str.split("INTEGER:")[1].strip())
        elif val_str.startswith("STRING:"):
            str_val = val_str.split("STRING:")[1].strip().strip('"')
            val = 0 if str_val == "" else float(str_val)
        elif val_str.startswith("Counter32:") or val_str.startswith("Gauge32:"):
            val = float(val_str.split(":")[1].strip())
        else:
            try:
                val = float(val_str.split()[0])
            except ValueError:
                val = 0.0
        return val, response_time
    except subprocess.TimeoutExpired:
        print(f"Timeout SNMP pour OID {oid}")
        return 0.0, float('inf')
    except subprocess.CalledProcessError as e:
        print(f"Erreur SNMP pour OID {oid}: {e.stderr}")
        return 0.0, 0.0
    except Exception as e:
        print(f"Erreur inattendue pour OID {oid}: {str(e)}")
        return 0.0, 0.0

def collect_snmp_data():
    data = {}
    total_response_time = 0
    valid_responses = 0
    for name, oid in OIDS.items():
        value, rt = snmp_get(oid)
        data[name] = value
        if rt != float('inf'):
            total_response_time += rt
            valid_responses += 1
    avg_response_time = total_response_time / valid_responses if valid_responses > 0 else float('inf')
    data["avg_response_time"] = avg_response_time
    return data, avg_response_time

def send_alert(subject, message):
    if not EMAIL_ENABLED:
        return
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = ALERT_EMAIL
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print("Alerte envoyée par email")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email: {e}")

def train_model(samples):
    X_train = np.array(samples)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train)
    joblib.dump(model, MODEL_FILE)
    print("Modèle entraîné et sauvegardé")
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        # Vérifier que le modèle correspond au nombre actuel de features
        if model.n_features_in_ != len(OIDS) + 1:  # +1 pour avg_response_time
            print("Le modèle existant ne correspond pas au nombre de features actuel.")
            print("Suppression du modèle pour réentraînement.")
            os.remove(MODEL_FILE)
            return None
        return model
    return None

def check_anomalies(model, data, threshold=-0.2):
    sample = np.array(list(data.values())).reshape(1, -1)
    score = model.decision_function(sample)[0]
    prediction = model.predict(sample)[0]
    print(f"Score décision: {score:.3f}")
    if prediction == -1 and score < threshold:
        print(f"ANOMALIE DETECTEE! Score: {score:.2f}")
        return True, score
    return False, score

def export_to_csv():
    """Exporte les données collectées vers des fichiers CSV"""
    if not collected_data:
        print("Aucune donnée à exporter")
        return
    
    try:
        # Définir les en-têtes de colonnes
        if collected_data:
            headers = ['date_heure', 'timestamp', 'probleme_detecte', 'type_probleme', 'score_anomalie'] + \
                     [key for key in collected_data[0].keys() if key not in ['timestamp', 'probleme_detecte', 'type_probleme', 'score_anomalie']]
        
        # Exporter les données principales
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for row in sorted(collected_data, key=lambda x: x['timestamp']):
                # Ajouter la date formatée
                formatted_row = row.copy()
                formatted_row['date_heure'] = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Réorganiser selon l'ordre des headers
                ordered_row = {header: formatted_row.get(header, '') for header in headers}
                writer.writerow(ordered_row)
        
        # Créer les statistiques
        total_samples = len(collected_data)
        anomalies = len([d for d in collected_data if d['probleme_detecte'] == 'Oui'])
        offline_count = len([d for d in collected_data if d['type_probleme'] == 'Appareil hors ligne'])
        anomaly_percentage = (anomalies / total_samples * 100) if total_samples > 0 else 0
        
        if total_samples > 0:
            first_time = datetime.fromtimestamp(min(d['timestamp'] for d in collected_data)).strftime('%Y-%m-%d %H:%M:%S')
            last_time = datetime.fromtimestamp(max(d['timestamp'] for d in collected_data)).strftime('%Y-%m-%d %H:%M:%S')
            duration_hours = (max(d['timestamp'] for d in collected_data) - min(d['timestamp'] for d in collected_data)) / 3600
        else:
            first_time = last_time = "N/A"
            duration_hours = 0
        
        # Exporter les statistiques
        stats_data = [
            ['Statistique', 'Valeur'],
            ['Nombre total d\'échantillons', total_samples],
            ['Nombre d\'anomalies détectées', anomalies],
            ['Nombre d\'appareils hors ligne', offline_count],
            ['Pourcentage d\'anomalies', f"{anomaly_percentage:.2f}%"],
            ['Première mesure', first_time],
            ['Dernière mesure', last_time],
            ['Durée de surveillance (heures)', f"{duration_hours:.2f}"]
        ]
        
        with open(STATS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(stats_data)
        
        print(f"Données exportées vers {CSV_FILE} ({total_samples} lignes)")
        print(f"Statistiques exportées vers {STATS_FILE}")
        
    except Exception as e:
        print(f"Erreur lors de l'export CSV: {e}")

def add_data_to_collection(data, has_problem, problem_type, anomaly_score):
    """Ajoute une nouvelle ligne de données à la collection"""
    timestamp = time.time()
    row = {
        'timestamp': timestamp,
        'probleme_detecte': 'Oui' if has_problem else 'Non',
        'type_probleme': problem_type if has_problem else 'Aucun',
        'score_anomalie': anomaly_score
    }
    
    # Ajouter toutes les métriques SNMP
    for key, value in data.items():
        row[key] = value
    
    collected_data.append(row)

def monitor():
    model = load_model()
    if model is None:
        print(f"Collecte initiale de {TRAINING_SAMPLES} échantillons...")
        for _ in range(TRAINING_SAMPLES):
            data, _ = collect_snmp_data()
            data_buffer.append(list(data.values()))
            # Ajouter les données d'entraînement à la collection
            add_data_to_collection(data, False, 'Phase d\'entraînement', 0.0)
            time.sleep(SAMPLE_INTERVAL)
        model = train_model(list(data_buffer))
    
    print("Démarrage de la surveillance...")
    last_retrain = time.time()
    last_export = time.time()
    
    while True:
        data, avg_response_time = collect_snmp_data()
        print("Données collectées:", data)
        
        has_problem = False
        problem_type = 'Aucun'
        anomaly_score = 0.0
        
        if avg_response_time == float('inf'):
            print("APPAREIL HORS LIGNE!")
            has_problem = True
            problem_type = 'Appareil hors ligne'
            anomaly_score = float('-inf')
            send_alert(
                "Appareil SNMP hors ligne",
                f"L'appareil {SNMP_HOST} ne répond plus. Temps de réponse dépassé."
            )
        else:
            is_anomaly, score = check_anomalies(model, data)
            anomaly_score = score
            if is_anomaly:
                has_problem = True
                problem_type = 'Anomalie détectée'
                send_alert(
                    "Alerte Anomalie SNMP",
                    f"Anomalie détectée sur {SNMP_HOST}\n\n" +
                    "\n".join(f"{k}: {v}" for k, v in data.items())
                )
        
        # Ajouter les données à la collection
        add_data_to_collection(data, has_problem, problem_type, anomaly_score)
        
        data_buffer.append(list(data.values()))
        
        # Réentraînement périodique
        if time.time() - last_retrain > RETRAIN_INTERVAL:
            print("Réentraînement du modèle avec nouvelles données...")
            model = train_model(list(data_buffer))
            last_retrain = time.time()
        
        # Export CSV périodique
        if time.time() - last_export > EXPORT_INTERVAL:
            print("Export des données vers CSV...")
            export_to_csv()
            last_export = time.time()
        
        time.sleep(SAMPLE_INTERVAL)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("Surveillance arrêtée")
        # Export final avant fermeture
        print("Export final des données...")
        export_to_csv()
        print("Fin du programme")