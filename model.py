from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


class MyModel:

    def __init__(self, model_name: str):
        self.model = self.load_model(model_name)

    @staticmethod
    def create_model():
        X, y = load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model: LogisticRegression = LogisticRegression(random_state=42, max_iter=80)
        model.fit(x_train, y_train)
        acc = accuracy_score(y_test, model.predict(x_test))
        print(f"model accuracy: {acc * 100:.2f}%")
        return model

    def save_model(self, file_name: str) -> None:
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    def load_model(self, file_name: str) -> LogisticRegression:
        try:
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
        except FileNotFoundError:
            model = self.create_model()
            self.save_model(file_name)
        return model

    if __name__ == '__main__':
        filename: str = "model.pkl"
        model = create_model()
        save_model(model, filename)
