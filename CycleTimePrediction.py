# Knihovny
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, make_scorer
# Knihovna pro použití grafů
import matplotlib.pyplot as plt
# Knihovna pro použití ODBC driveru
import pyodbc
# Knihovna pro použití tabulek
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    filtered_columns = [col for col in df_cleaned.columns if 'WGHT' in col or 'Nazev' in col or col == 'DelkaTrvani' or col == 'Program' or col == 'Pondeli']
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funkce pro odstranení odlehlých hodnot
def remove_outliers(df):
    # Definice vstupních a cílových proměnných pro detekci odlehlých hodnot
    feature_columns = ['Total_WGHT', 'Program', 'Pondeli'] + [col for col in df.columns if col.startswith("Nazev_")]
    X = df[feature_columns]
    y = df['DelkaTrvani']
    
    # Fit lineární regresi pro získání předpovědí
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Vypočítat rezidua (rozdíl mezi skutečnými a predikovanými hodnotami)
    residuals = np.abs(y - y_pred)
    
    # Stanovit hranici pro odlehlé hodnoty (například 1.5x interkvartilové rozpětí)
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR  # Hranice pro odlehlé hodnoty
    
    # Filtrace dat bez odlehlých hodnot
    df_filtered = df[residuals <= threshold]
    
    return df_filtered

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funkce pro týkající se strojového učení
def machine_learning(df_finished, df_unfinished):

    #print(df_finished)

    # One Hot Encoding pro sloupec 'Nazev'
    ohe_nazev = pd.get_dummies(df_finished['Nazev']).astype('float64')
    df_prepared = pd.concat([df_finished.drop(columns=['Nazev']).reset_index(drop=True), ohe_nazev.reset_index(drop=True)], axis=1)

    # Funkce, která odstraní hodnoty, které jsou příliš odlehlé
    df_prepared = remove_outliers(df_prepared)

    #print(df_prepared)

    # Korelační matice
    # correlation_matrix = df_prepared.corr()
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    # plt.title('Korelační matice mezi vstupy a cílem (DelkaTrvani)')
    # plt.show()

    # Definice vstupních a cílových proměnných pro trénink modelu
    X = df_prepared[['Total_WGHT', 'Program', 'Pondeli'] + list(ohe_nazev.columns)]
    y = df_prepared['DelkaTrvani']

    # Standardizace vstupních dat
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Trénink lineárního regresního modelu
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Parametry pro filtrování a predikci
    total_weight = 3000  # zadaná hodnota
    program = 5         # zadaná hodnota
    pondeli = 0          # zadaná hodnota
    nazev = ['VD']  # název pro One Hot Encoding

    # Připravit vstupní data v souladu s OHE sloupci
    input_data = pd.DataFrame({
        'Total_WGHT': [total_weight],
        'Program': [program],
        'Pondeli': [pondeli]
    })

    for col in ohe_nazev.columns:
        input_data[col] = [1.0 if col in nazev else 0.0]

    # Standardizace vstupních dat pro predikci
    input_data_scaled = scaler.transform(input_data)

    # Predikce
    prediction = model.predict(input_data_scaled)
    print(f"Předpovězená DelkaTrvani pro zadané parametry: {prediction[0]}")

    # # Vizualizace
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=df_prepared['Total_WGHT'], y=df_prepared['DelkaTrvani'], label="Data")
    
    # # Přidání regresní přímky
    # sns.regplot(x=df_prepared['Total_WGHT'], y=df_prepared['DelkaTrvani'], scatter=False, color="red", label="Regresní křivka")
    
    # #Přidání predikovaného bodu
    # plt.scatter(total_weight, prediction[0], color='blue', label="Predikovaná hodnota", marker="X", s=100)
    
    # plt.xlabel('Total_WGHT')
    # plt.ylabel('DelkaTrvani')
    # plt.title('Závislost DelkaTrvani na Total_WGHT s predikovanou hodnotou')
    # plt.legend()
    # plt.show()

    nazev = 'VD'

    # Volání funkce
    machine_learning_filtered(df_finished, program, pondeli, nazev, total_weight)
    machine_learning_knn(df_prepared, ohe_nazev, total_weight, program, pondeli, nazev,5)

    return df_finished, df_unfinished

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def machine_learning_knn(df_prepared, ohe_nazev, total_weight, program, pondeli, nazev, k):

    # Reset indexu pro konzistentní zobrazení
    df_prepared = df_prepared.reset_index(drop=True)

    print(df_prepared)
    
    # Define features and target variables for the KNN model
    X = df_prepared[['Total_WGHT', 'Program', 'Pondeli'] + list(ohe_nazev.columns)]
    y = df_prepared['DelkaTrvani']
    
    # Set k to the maximum possible if k is larger than the number of samples
    k = min(k, len(X))
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize KNN model
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_scaled, y)
    
    # Prepare input data for prediction, ensuring it has all the columns from X
    input_data = pd.DataFrame({
        'Total_WGHT': [total_weight],
        'Program': [program],
        'Pondeli': [pondeli]
    })
    for col in ohe_nazev.columns:
        input_data[col] = [1.0 if col == nazev else 0.0]
    
    input_data_scaled = scaler.transform(input_data)
    
    # Predikce pomocí průměru K nejbližších sousedů
    prediction = knn_model.predict(input_data_scaled)
    print(f"Předpovězená DelkaTrvani pro zadané parametry (pomocí KNN s k={k}): {prediction[0]}")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def machine_learning_filtered(df_finished, program, pondeli, nazev, total_weight):

    # Filtrace dat podle `Program`, `Pondeli` a `Nazev`
    filter_condition = (
        (df_finished['Program'] == program) &
        (df_finished['Pondeli'] == pondeli) &
        (df_finished['Nazev'] == nazev)
    )
    df_filtered = df_finished[filter_condition].copy()

    # Kontrola, zda filtrace vrátila nějaká data
    if df_filtered.empty:
        print("Žádná data nesplňují zadané podmínky filtrování.")
        return None

    # Definice vstupních a cílových proměnných
    X = df_filtered[['Total_WGHT']]
    y = df_filtered['DelkaTrvani']

    # Standardizace vstupních dat
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Trénink lineárního regresního modelu
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Příprava vstupního data pro predikci
    input_data = pd.DataFrame({'Total_WGHT': [total_weight]})
    input_data_scaled = scaler.transform(input_data)

    # # Predikce
    prediction = model.predict(input_data_scaled)
    print(f"Předpovězená DelkaTrvani pro filtrovane parametry: {prediction[0]}")

    # # Vizualizace
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=df_filtered['Total_WGHT'], y=df_filtered['DelkaTrvani'], label="Data")
    
    # # Přidání regresní přímky
    # sns.regplot(x=df_filtered['Total_WGHT'], y=df_filtered['DelkaTrvani'], scatter=False, color="red", label="Regresní křivka")
    
    # # Přidání predikovaného bodu
    # plt.scatter(total_weight, prediction[0], color='blue', label="Predikovaná hodnota", marker="X", s=100)
    
    # plt.xlabel('Total_WGHT')
    # plt.ylabel('DelkaTrvani')
    # plt.title(f'Závislost DelkaTrvani na Total_WGHT pro Program={program}, Pondeli={pondeli}, Nazev={nazev}')
    # plt.legend()
    # plt.show()

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

# Cekani na vstup
input()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------