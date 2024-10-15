# fastapi-ml-doctor

Ce projet, intitulé **fastapi-ml-doctor**, vise à créer une API capable de prédire des maladies à partir de symptômes fournis par l'utilisateur. Il utilise **FastAPI**, un framework web Python performant et moderne, pour déployer un modèle de machine learning (ML) qui a été préalablement entraîné. Voici un résumé du projet en français :

### Contexte du projet

L'objectif principal est de construire une application de médecin virtuel alimentée par l'intelligence artificielle (IA) qui peut diagnostiquer des maladies en fonction des symptômes fournis. Le projet met en œuvre un modèle de régression logistique développé avec la bibliothèque **scikit-learn**. Le modèle est pré-entraîné et disponible sur la plateforme **Hugging Face Hub**, une plateforme centralisée qui héberge des modèles ML prêts à l'emploi.

### Étapes clés du projet

1. **Téléchargement et installation des dépendances** :
   - Le projet nécessite l'installation de plusieurs bibliothèques, dont **FastAPI**, **scikit-learn** (pour la gestion des modèles ML), et **huggingface_hub** (pour télécharger le modèle depuis Hugging Face).
   - Commande d'installation : 
     ```bash
     pip install fastapi[all] scikit-learn huggingface_hub
     ```

2. **Téléchargement du modèle ML** :
   - Le modèle utilisé pour les prédictions s'appelle **human-disease-prediction**, un modèle léger de régression logistique développé avec **scikit-learn**.
   - Ce modèle est stocké sur **Hugging Face Hub**, et est téléchargé directement via l'API Python.

3. **Architecture du projet** :
   - Le projet contient deux fichiers principaux :
     - **utils.py** : Ce fichier contient la liste des symptômes (*symptoms_list*) que le modèle accepte comme entrée.
     - **main.py** : C'est le fichier principal de l'application qui contient le code de l'API FastAPI. Il intègre le modèle en utilisant la fonctionnalité **lifespan** de FastAPI, qui permet de gérer le cycle de vie de l'application (chargement du modèle au démarrage et nettoyage à la fermeture).

4. **Lancer le serveur FastAPI** :
   - Une fois le code en place, vous pouvez démarrer l'API en exécutant la commande suivante :
     ```bash
     uvicorn app.main:app
     ```
   - L'API sera accessible localement à l'adresse : [http://localhost:8000/docs](http://localhost:8000/docs). Cette interface interactive permet de tester facilement les différentes requêtes API, comme celle pour diagnostiquer une maladie.

### Fonctionnement de l'API

L'API propose une route `/diagnosis` qui permet à l'utilisateur de fournir une liste de symptômes (sous forme de booléens), et en réponse, l'API retourne une prédiction de maladie. Le modèle de régression logistique utilise cette liste pour générer une prédiction.

En résumé, ce projet illustre comment déployer un modèle de machine learning en production à l'aide de FastAPI et **Joblib** (pour la sérialisation et désérialisation des modèles). Le modèle est léger et utilise une approche de régression logistique pour prédire des maladies à partir de symptômes spécifiques fournis par l'utilisateur.



FastAPI provides a robust framework for building web services, making it an ideal choice for deploying ML models in production environments. In this recipe, we will see how to integrate an ML model with FastAPI using Joblib, a popular library for model serialization and deserialization in Python.

We will develop an AI-powered doctor application that can diagnose diseases by analyzing the symptoms provided.

pip install fastapi[all] scikit-learn

We will download the model from the Hugging Face Hub, a centralized hub hosting pre-trained ML models that are ready to be used.

We will use the human-disease-prediction model, which is a relatively lightweight linear logistic regression model developed with the scikit-learn package. You can check it out at the following link: https://huggingface.co/AWeirdDev/human-disease-prediction.

pip install huggingface_hub

## How to do it 

1. Let’s start by writing the code to accommodate the ML model. In the project root folder, let's create the app folder containing a module called utils.py. In the module, we will declare a symptoms_list list containing all the symptoms accepted by the model. 

2. Still in the app folder, let’s create the main.py module that will contain the FastAPI server class object and the endpoint. To incorporate the model into our application, we will utilize the FastAPI lifespan feature.

3. Write the code of the api in the main.py file

4. uvicorn app.main:app

5. http://localhost:8000/docs