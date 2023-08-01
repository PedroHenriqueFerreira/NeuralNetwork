from database import Database
# from preprocessing import LabelEncoder, OneHotEncoder

# hot_encoder = OneHotEncoder()
# label_encoder = LabelEncoder()

db = Database.read_csv('test.csv')

print(db)