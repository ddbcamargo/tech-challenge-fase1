"""Aplicação Flask para servir o melhor modelo de ML.

Responsabilidades:
- Criar a aplicação Flask.
- Registrar blueprints de rotas (``health`` e ``predict``).
- Registrar a UI do Swagger em ``/apidocs/``.
- Delegar toda a lógica de inferência para :mod:`src.api.services`.

Execução:
    python -m src.api.app

Links úteis (após subir a API):
    - Swagger UI:   http://localhost:5000/apidocs/
    - OpenAPI JSON: http://localhost:5000/apispec.json
"""
from __future__ import annotations

from flasgger import Flasgger
from flask import Flask, jsonify, redirect

from src.api.docs.openapi import SWAGGER_CONFIG, build_openapi_spec
from src.api.routes.health import health_bp
from src.api.routes.predict import predict_bp


def create_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(health_bp)
    app.register_blueprint(predict_bp)

    Flasgger(app, template=build_openapi_spec(), config=SWAGGER_CONFIG)

    @app.get("/")
    def root():
        # Redireciona a raiz para o Swagger UI — facilita o teste manual.
        return redirect("/apidocs/", code=302)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Rota não encontrada."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify({"error": "Método HTTP não permitido."}), 405

    @app.errorhandler(500)
    def internal_error(_):
        return jsonify({"error": "Erro interno no servidor."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
