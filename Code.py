from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import numpy as np
# Knihovna pro použití grafů
import matplotlib.pyplot as plt
# Knihovna pro použití ODBC driveru
import pyodbc
# Knihovna pro použití tabulek
import pandas as pd

# Nastavení pro zobrazení celé tabulky bez omezení řádků a sloupců
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funkce, která upraví SQL tabulku na a vyfiltruje nepotřebná data
def get_filtered_data(df):

     # Vynechání řádků, kde je ve sloupcích 'Batch01_GIN', 'Batch01_Lot', 'Batch01_WGHT', 'Batch01_stred', 'Batch01_cont', 'Batch01_Nazev', 'Batch01_CisloZakazky' hodnota 0 nebo NULL (NaN)
    required_columns = ['Batch01_GIN', 'Batch01_Lot', 'Batch01_WGHT', 'Batch01_stred', 'Batch01_cont', 'Batch01_Nazev', 'Batch01_CisloZakazky']
    for col in required_columns:
        df = df[(df[col] != 0) & (df[col].notna())]

    # Vynechání řádků, kde je ve sloupci 'CasZahajeni' hodnota 1900-01-01 00:00:00
    df = df[df['CasZahajeni'] != pd.Timestamp('1900-01-01 00:00:00')]

    # Export CSV

    # Vytvoření druhé tabulky pro záznamy, kde je CasUkonceni NaN nebo 1900-01-01 nebo kde je StatusProgramu jiný než 0
    df_unfinished = df[(df['CasUkonceni'].isna()) | 
                       (df['CasUkonceni'] == pd.Timestamp('1900-01-01 00:00:00')) | 
                       (df['StatusProgramu'] != 0)] 

    # Odstranění problémových záznamů z původní tabulky
    df_cleaned = df.drop(df_unfinished.index)

    # Zanechání pouze řádků, kde je 'StatusProgramu' rovno 1
    df_unfinished = df_unfinished[df_unfinished['StatusProgramu'] == 1]

    # Zanechání pouze sloupců 'ID', 'WGHT' a 'Nazev'
    df_unfinished = df_unfinished[['ID'] + [col for col in df_unfinished.columns if 'WGHT' in col or 'Nazev' in col]]

    # Vytvoření nového sloupce pro výpočet délky trvání (rozdíl mezi 'CasZahajeni' a 'CasUkonceni')
    df_cleaned.insert(0, 'DelkaTrvani', (df_cleaned['CasUkonceni'] - df_cleaned['CasZahajeni']).dt.total_seconds().astype(int))  # Převedení na celá čísla  

    #########################################################################################################################################################################################
    df_cleaned = pd.read_csv('clenad.csv',sep=';')
    #########################################################################################################################################################################################

    # Vytvoř nový sloupec a do neho zapiš 1, pokud CasZahajeni je pondeli, jinak 0
    df_cleaned['CasZahajeni'] = pd.to_datetime(df_cleaned['CasZahajeni'], dayfirst=True)
    df_cleaned['Pondeli'] = (df_cleaned['CasZahajeni'].dt.dayofweek == 0).astype(int)

    #display(df_cleaned)

    # Vynechání sloupců 'CasZahajeni' a 'CasUkonceni'
    #df_cleaned = df_cleaned.drop(columns=['CasZahajeni', 'CasUkonceni'])

    # Omezení DataFrame na sloupce, které obsahují 'WGHT', 'Nazev', 'DelkaTrvani' nebo 'ID'
    filtered_columns = [col for col in df_cleaned.columns if 'WGHT' in col or 'Nazev' in col or col == 'DelkaTrvani' or col == 'ID' or col == 'Program' or col == 'Pondeli']
    df_filtered = df_cleaned[filtered_columns]

    # Vynechání řádků, kde je hodnota ve sloupci 'DelkaTrvani' menší než 3600 sekund (1 hodina)
    df_filtered = df_filtered[df_filtered['DelkaTrvani'] >= 3600]

    # Nahrazení hodnot NaN ve sloupcích s hmotností hodnotou 0 pro všechny relevantní sloupce
    for col in df_filtered.columns:
        if 'WGHT' in col:
            df_filtered[col] = df_filtered[col].fillna(0)

    # Vytvoření nového sloupce pro celkovou váhu všech "WGHT" sloupců
    wght_columns = [col for col in df_filtered.columns if 'WGHT' in col]
    df_filtered['Total_WGHT'] = df_filtered[wght_columns].sum(axis=1)

    # Přesunutí sloupce 'Total_WGHT' na druhou pozici
    cols = list(df_filtered.columns)
    cols.insert(1, cols.pop(cols.index('Total_WGHT')))
    df_filtered = df_filtered[cols]

    # Odstranění původních "WGHT" sloupců
    df_filtered = df_filtered.drop(columns=wght_columns)

    # Vymazani sloupce ID
    df_filtered = df_filtered.drop(columns=['ID'])

    # Přenechání sloupců 'Total_WGHT' a 'Program' atd..
    df_filtered = df_filtered[['Total_WGHT', 'Program', 'DelkaTrvani', 'Batch01_Nazev', 'Pondeli']]

    # Vymazání řádků, kde chybí hodnota ve sloupci 'Program'
    df_filtered = df_filtered.dropna(subset=['Program'])

    # Přejmenování sloupce 'Batch01_Nazev' na 'Nazev'
    df_finished = df_filtered.rename(columns={"Batch01_Nazev": "Nazev"})

    # Export df_filtered do CSV
    df_finished.to_csv('df_finihed.csv', index=False)

    return df_finished, df_unfinished 

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Připojení k databázi pomocí ODBC driveru
def SQL_connection():
    try:
        connection = pyodbc.connect(
            'DRIVER={SQL Server};'
            r'SERVER=1VYVOJ1-VM\SQLEXPRESS;' # Název serveru
            'DATABASE=ESABHistorian;' # Název databáze
            'Trusted_Connection=yes;'
        )
        print("Connection successful to ESABHistorian.\n")
    except pyodbc.Error as e:
        print("Connection failed to ESABHistorian.",e, "\n")

    # Připojení k databázi
    cursor = connection.cursor()

    # Provádění SQL dotazu
    cursor.execute("SELECT * FROM dbo.Davky")

    df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
    connection.close()
    return df

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Načtení dat z CSV
def read_data_from_csv():
    csv_file_path = 'export_tabulky_davky_241014.csv'
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Data byla nactena z CSV: {csv_file_path}.\n")
    except FileNotFoundError:    
        print(f"Soubor {csv_file_path} nebyl nalezen.\n")
        return None

    # Převod sloupců CasZahajeni a CasUkonceni na datetime
    df['CasZahajeni'] = pd.to_datetime(df['CasZahajeni'], errors='coerce')
    df['CasUkonceni'] = pd.to_datetime(df['CasUkonceni'], errors='coerce')

    return df

# Funkce pro týkající se strojového učení
def machine_learning(df_finished, df_unfinished):

    # One Hot Encoding pro Nazev
    ohe_nazev = pd.get_dummies(df_finished['Nazev'])

    ohe_nazev = ohe_nazev.astype(int)

    X = pd.concat([df_finished[['Total_WGHT', 'Program','Pondeli']], ohe_nazev], axis=1)
    y = df_finished['DelkaTrvani']                
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Display evaluation metrics
    print(f"Decision Tree Model")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print("\n")
    
    # Show some of the predicted vs actual values
    comparison_df = pd.DataFrame({'Predicted': y_pred[:10], 'Actual': y_test[:10].values})
    print(comparison_df)

    return df_finished, df_unfinished

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Hlavní část programu

# Načtení dat z CSV
df = read_data_from_csv()

# Funkce pro týkající se filtrovaní dat
df_finished, df_unfinished = get_filtered_data(df)

# Funkce pro týkající se strojového učení
df_finished, df_unfinished = machine_learning(df_finished, df_unfinished)

#Vykreslení grafu
#plot_graph(df_finished)

input("Stiskněte Enter pro ukončení programu...")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------