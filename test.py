from database import Database
from preprocessing import LabelEncoder, OneHotEncoder

hot_encoder = OneHotEncoder()

db = Database.read_csv('test.csv')

db[0:2] = [1,2,3,4,5,6]

print(db)