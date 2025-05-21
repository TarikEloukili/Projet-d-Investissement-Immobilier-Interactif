from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
import sqlite3
import os

app = Flask(__name__)

class RealEstatePredictor:
    def __init__(self):
        self.short_term_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.medium_term_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.long_term_model = RandomForestRegressor(n_estimators=150, random_state=42)
        self.scaler = StandardScaler()
        self.le_zone = LabelEncoder()
        self.le_type = LabelEncoder()
        self.is_trained = False
        
    def generate_sample_data(self, n_samples=1000):
        """Génère des données d'exemple pour l'entraînement"""
        np.random.seed(42)
        
        zones = ['Centre-ville', 'Banlieue', 'Périphérie', 'Zone industrielle', 'Quartier résidentiel']
        types_biens = ['Appartement', 'Maison', 'Studio', 'Duplex', 'Villa']
        
        data = []
        for _ in range(n_samples):
            zone = np.random.choice(zones)
            type_bien = np.random.choice(types_biens)
            surface = np.random.normal(80, 30)
            surface = max(20, surface)  # Surface minimum
            
            # Facteurs d'influence du prix
            zone_multiplier = {
                'Centre-ville': 1.5,
                'Banlieue': 1.2,
                'Périphérie': 0.8,
                'Zone industrielle': 0.7,
                'Quartier résidentiel': 1.0
            }[zone]
            
            type_multiplier = {
                'Appartement': 1.0,
                'Maison': 1.3,
                'Studio': 0.7,
                'Duplex': 1.4,
                'Villa': 1.8
            }[type_bien]
            
            chambres = np.random.randint(1, 6)
            age = np.random.randint(0, 50)
            proximite_transport = np.random.uniform(0.1, 2.0)  # km
            note_quartier = np.random.uniform(1, 5)
            
            # Prix de base calculé
            prix_base = (surface * 3000 * zone_multiplier * type_multiplier * 
                        (1 - age * 0.01) * (1 + note_quartier * 0.1) * 
                        (1 / (1 + proximite_transport * 0.1)))
            
            # Ajout de bruit
            prix_actuel = prix_base * np.random.normal(1, 0.1)
            prix_actuel = max(50000, prix_actuel)
            
            # Prédictions futures avec tendances
            croissance_court = np.random.normal(0.03, 0.02)  # 3% ± 2%
            croissance_moyen = np.random.normal(0.05, 0.03)  # 5% ± 3% par an
            croissance_long = np.random.normal(0.04, 0.02)   # 4% ± 2% par an
            
            prix_1an = prix_actuel * (1 + croissance_court)
            prix_5ans = prix_actuel * ((1 + croissance_moyen) ** 5)
            prix_20ans = prix_actuel * ((1 + croissance_long) ** 20)
            
            data.append({
                'zone': zone,
                'type_bien': type_bien,
                'surface': surface,
                'chambres': chambres,
                'age': age,
                'proximite_transport': proximite_transport,
                'note_quartier': note_quartier,
                'prix_actuel': prix_actuel,
                'prix_1an': prix_1an,
                'prix_5ans': prix_5ans,
                'prix_20ans': prix_20ans
            })
        
        return pd.DataFrame(data)
    
    def train_models(self):
        """Entraîne les modèles de prédiction"""
        # Génération des données d'entraînement
        df = self.generate_sample_data(2000)
        
        # Préparation des features
        df['zone_encoded'] = self.le_zone.fit_transform(df['zone'])
        df['type_encoded'] = self.le_type.fit_transform(df['type_bien'])
        
        features = ['surface', 'chambres', 'age', 'proximite_transport', 
                   'note_quartier', 'zone_encoded', 'type_encoded']
        
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement des trois modèles
        y_short = df['prix_1an']
        y_medium = df['prix_5ans'] 
        y_long = df['prix_20ans']
        
        self.short_term_model.fit(X_scaled, y_short)
        self.medium_term_model.fit(X_scaled, y_medium)
        self.long_term_model.fit(X_scaled, y_long)
        
        self.is_trained = True
        
        # Sauvegarde des modèles
        joblib.dump(self.short_term_model, 'models/short_term_model.pkl')
        joblib.dump(self.medium_term_model, 'models/medium_term_model.pkl')
        joblib.dump(self.long_term_model, 'models/long_term_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.le_zone, 'models/le_zone.pkl')
        joblib.dump(self.le_type, 'models/le_type.pkl')
        
        return True
    
    def predict_prices(self, zone, type_bien, surface, chambres, age, proximite_transport, note_quartier):
        """Prédit les prix à court, moyen et long terme"""
        if not self.is_trained:
            return None
            
        try:
            # Préparation des données
            zone_encoded = self.le_zone.transform([zone])[0] if zone in self.le_zone.classes_ else 0
            type_encoded = self.le_type.transform([type_bien])[0] if type_bien in self.le_type.classes_ else 0
            
            features = np.array([[surface, chambres, age, proximite_transport, 
                                note_quartier, zone_encoded, type_encoded]])
            features_scaled = self.scaler.transform(features)
            
            # Prédictions
            prix_court = self.short_term_model.predict(features_scaled)[0]
            prix_moyen = self.medium_term_model.predict(features_scaled)[0]
            prix_long = self.long_term_model.predict(features_scaled)[0]
            
            return {
                'prix_1an': round(prix_court, 0),
                'prix_5ans': round(prix_moyen, 0),
                'prix_20ans': round(prix_long, 0)
            }
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            return None

# Initialisation du prédicteur
predictor = RealEstatePredictor()

# Création du dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Données des zones pour la carte
zones_data = [
    {"id": 1, "nom": "Centre-ville", "lat": 48.8566, "lng": 2.3522, "prix_moyen": 8500, "potentiel": 85},
    {"id": 2, "nom": "Banlieue", "lat": 48.8766, "lng": 2.3722, "prix_moyen": 6200, "potentiel": 75},
    {"id": 3, "nom": "Périphérie", "lat": 48.8366, "lng": 2.3322, "prix_moyen": 4800, "potentiel": 90},
    {"id": 4, "nom": "Zone industrielle", "lat": 48.8166, "lng": 2.3122, "prix_moyen": 3200, "potentiel": 70},
    {"id": 5, "nom": "Quartier résidentiel", "lat": 48.8666, "lng": 2.3922, "prix_moyen": 5500, "potentiel": 80},
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/zones')
def get_zones():
    return jsonify(zones_data)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Entraînement du modèle si nécessaire
        if not predictor.is_trained:
            predictor.train_models()
        
        predictions = predictor.predict_prices(
            zone=data['zone'],
            type_bien=data['type_bien'],
            surface=float(data['surface']),
            chambres=int(data['chambres']),
            age=int(data['age']),
            proximite_transport=float(data['proximite_transport']),
            note_quartier=float(data['note_quartier'])
        )
        
        if predictions:
            # Calcul des rendements potentiels
            prix_actuel_estime = predictions['prix_1an'] * 0.97  # Estimation prix actuel
            rendement_1an = ((predictions['prix_1an'] - prix_actuel_estime) / prix_actuel_estime) * 100
            rendement_5ans = ((predictions['prix_5ans'] - prix_actuel_estime) / prix_actuel_estime / 5) * 100
            rendement_20ans = ((predictions['prix_20ans'] - prix_actuel_estime) / prix_actuel_estime / 20) * 100
            
            result = {
                'predictions': predictions,
                'prix_actuel_estime': round(prix_actuel_estime, 0),
                'rendements': {
                    'court_terme': round(rendement_1an, 2),
                    'moyen_terme': round(rendement_5ans, 2),
                    'long_terme': round(rendement_20ans, 2)
                },
                'recommandation': get_recommendation(rendement_1an, rendement_5ans, rendement_20ans)
            }
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Erreur lors de la prédiction'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_recommendation(rend_1an, rend_5ans, rend_20ans):
    """Génère une recommandation d'investissement"""
    if rend_5ans > 8 and rend_20ans > 6:
        return {
            'niveau': 'Excellent',
            'couleur': 'success',
            'texte': 'Investissement très attractif avec un fort potentiel de croissance.'
        }
    elif rend_5ans > 5 and rend_20ans > 4:
        return {
            'niveau': 'Bon',
            'couleur': 'warning',
            'texte': 'Bon investissement avec un potentiel de croissance correct.'
        }
    else:
        return {
            'niveau': 'Moyen',
            'couleur': 'secondary',
            'texte': 'Investissement modéré, à considérer avec prudence.'
        }

@app.route('/api/zone-analysis/<int:zone_id>')
def get_zone_analysis(zone_id):
    """Analyse détaillée d'une zone"""
    zone = next((z for z in zones_data if z['id'] == zone_id), None)
    if not zone:
        return jsonify({'error': 'Zone non trouvée'}), 404
    
    # Simulation d'analyses de marché
    analysis = {
        'zone': zone,
        'tendances': {
            'prix_evolution_1an': np.random.uniform(-5, 15),
            'demande': np.random.uniform(60, 95),
            'offre': np.random.uniform(40, 80)
        },
        'infrastructures': {
            'transports': np.random.randint(3, 10),
            'commerces': np.random.randint(4, 9),
            'ecoles': np.random.randint(2, 8),
            'hopitaux': np.random.randint(1, 5)
        },
        'projets_futurs': [
            'Nouvelle ligne de métro prévue en 2026',
            'Centre commercial en construction',
            'Rénovation urbaine du quartier'
        ][:np.random.randint(1, 4)]
    }
    
    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)