import pytest
from app import app


@pytest.fixture
def client():
    app.config['testing'] = True
    with app.test_client() as client:
        yield client


def test_predictions_not_none(client):
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.get('/predict', json=payload)
    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data is not None


def test_predictions_length(client):
    """
    Test 2 (na maksymalną ocenę 5): Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    # ze względu, że w przypadku mojej aplikacji po wywołaniu predykcji zwracane są dwie informacje - index i nazwa
    # predyktowanej klasy sprawdzam, czy odpowiedź składa się z dwóch pól

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.get('/predict', json=payload)
    preds = response.get_json()
    assert len(preds) == 2

    # dodatkowo spradzam czy zwracane są odpowiednie pola i ich typy
    assert "label-index" in preds
    assert "label-prediction" in preds
    assert isinstance(preds["label-index"], int)
    assert isinstance(preds["label-prediction"], str)


def test_predictions_value_range(client):
    """
    Test 3 (na maksymalną ocenę 5): Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie: Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.get('/predict', json=payload)
    response_data = response.get_json()
    label_index = response_data['label-index']
    possible_predictions = {0, 1, 2}
    assert label_index in possible_predictions


def test_model_accuracy(client):
    """
    Test 4 (na maksymalną ocenę 5): Sprawdza, czy model osiąga co najmniej 70% dokładności (przykładowy warunek, można dostosować do potrzeb).
    """
    # w moim przypadku sprawdzam, czy dokładność wynosi co najmniej 90%
    response = client.get('/accuracy')
    accuracy = response.get_json()
    assert accuracy >= .9
