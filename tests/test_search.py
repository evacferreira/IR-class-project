import pytest
# Isto assume que vais criar uma função chamada 'preprocess' no ficheiro nlp.py
from src.search.nlp import preprocess

def test_preprocess_basic():
    texto = "O Algoritmo de Pesquisa no RepositóriUM!"
    resultado = preprocess(texto)
    
    # O que esperamos: minúsculas, sem "o", "de", "no" (stop words) e sem "!"
    # O resultado deve ser uma lista de palavras (tokens)
    assert "algoritmo" in resultado
    assert "pesquisa" in resultado
    assert "repositorium" in resultado
    assert "o" not in resultado  # Stop word removida
    assert "!" not in resultado  # Pontuação removida

def test_preprocess_empty():
    # Testar se o código aguenta um texto vazio sem crashar
    assert preprocess("") == []