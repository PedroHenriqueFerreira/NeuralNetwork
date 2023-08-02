from preprocessing import Normalizer, Standardizer

norm = Normalizer()

n = norm.fit_transform([[1, 2, 3], [4, 5, 6]])

print(n)

print(norm.inverse_transform(n))