from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


class MyModel:

    def __init__(self, model_name: str):
        self.model: LogisticRegression = MyModel.load_model(model_name)
        self.model_name = model_name
        self.accuracy = 0.0

    @staticmethod
    def create_model() -> LogisticRegression:
        X, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model: LogisticRegression = LogisticRegression(random_state=42, max_iter=80)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        print(f"model accuracy: {accuracy * 100:.2f}%")
        return model

    def save_model(self) -> None:
        with open(self.model_name, 'wb') as file:
            pickle.dump(self.model, file)

    @staticmethod
    def load_model(file_name) -> LogisticRegression:
        try:
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
        except FileNotFoundError:
            model = MyModel.create_model()
            return model

    def get_accuracy(self):
        X, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        return accuracy_score(y_test, self.model.predict(x_test))


if __name__ == '__main__':
    filename: str = "model.pkl"
    model = MyModel(filename)
    model.save_model()
