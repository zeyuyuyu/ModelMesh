# src/model_manager.py

import numpy as np
import requests

class ModelManager:
    def __init__(self):
        self.models = {}
        self.peers = []
        self.weights = None

    def add_model(self, model_id, model):
        self.models[model_id] = model

    def add_peer(self, peer_url):
        self.peers.append(peer_url)

    def federate_models(self):
        # Fetch model weights from peers
        for peer in self.peers:
            try:
                response = requests.get(f"{peer}/model")
                peer_weights = response.json()
                self.weights = self._aggregate_weights(self.weights, peer_weights)
            except requests.exceptions.RequestException:
                print(f"Error fetching model from {peer}")

        # Update local models with federated weights
        for model_id, model in self.models.items():
            model.set_weights(self.weights)

    def _aggregate_weights(self, local_weights, peer_weights):
        if local_weights is None:
            return peer_weights

        for layer in range(len(local_weights)):
            local_weights[layer] = (local_weights[layer] + peer_weights[layer]) / 2

        return local_weights