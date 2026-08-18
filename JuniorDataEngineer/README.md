# Project README

## Part 1 - Data Cleaning

### Approach

Najskôr som si prečítal zadanie a porovnal ho s mojimi predchádzajúcimi
projektmi. Podobnú úlohu som už riešil vo svojom verejnom repozitári 
(https://github.com/JMartin123456/Road-to-DS)
v časti `03_data_cleaning`.

Na základe tejto skúsenosti som vedel, aký postup a štruktúru riešenia
použiť. ChatGPT som následne použil najmä na pomoc pri tvorbe a úprave
kódu podľa požiadaviek zadania.

Výsledný kód som následne spustil a skontroloval, či funguje správne.
Zároveň som prešiel jednotlivé časti kódu a upravil ich tak, aby
zodpovedali spôsobu, akým by som riešenie vytvoril sám.

## Výstup

Zoznam affectnutych buniek sa nachadza v:
`01_afftected_cells.txt`

---

## Part 2 - Data Analysis

### Assumptions

Pri tejto časti som postupoval podobne ako pri Part 1. Vychádzal som
zo skúseností z môjho predchádzajúceho projektu `02_sales_analyzer`,
kde som už pracoval s podobnou analýzou dát.

Predpokladal som, že stĺpec `Price` obsahuje číselné hodnoty a že
`Category` a `Product ID` môžem použiť na filtrovanie a zoskupovanie
dát.

Pred výpočtami som preto stĺpec `Price` previedol na numerický typ.
Pri filtrovaní kategórie Tools som použil Product ID od 1030 do 1040
vrátane.

Výsledky analýzy som uložil do samostatného súboru
`data/02_analysis_results.csv`, aby pôvodný dataset
`assignment-database.csv` zostal nezmenený.

# Part 3 - Real Estate Data Scraper

## Popis

V tejto časti som vytvoril Python scraper pre webovú stránku Čerešne.
Cieľom bolo získať dostupné údaje o bytoch a uložiť ich do CSV súboru.

Scraper získava dáta zo stránky:

https://www.ceresne.sk/app/uploads/flats.json

Z dát získavam napríklad:
- ID bytu
- budovu
- poschodie
- počet izieb
- výmeru
- cenu
- stav bytu
- URL bytu

Dáta následne čistím a prevádzam na vhodné dátové typy. Vypočítavam
aj cenu za m².

## Výstup

Výsledný CSV súbor sa ukladá do:

`data/03_ceresne_listings.csv`

Scraper zároveň vykonáva základnú validáciu dát a vytvára tri jednoduché
grafy:

- počet bytov podľa statusu
- priemerná cena podľa počtu izieb
- rozdelenie cien bytov

Grafy sa ukladajú do priečinka:

`charts/`

## Validácia

Kontrolujem najmä:
- chýbajúce ID bytov
- chýbajúce ceny
- chýbajúce výmery
- duplicitné ID
- duplicitné URL
- počet bytov podľa statusu

## Optional Bonus

Z voliteľných bonusov som implementoval:

- **Basic data validation** – kontrola chýbajúcich hodnôt,
  duplicít a počtu bytov podľa statusu.
- **Small charts / summary** – vytvorenie troch základných grafov
  pomocou Matplotlib.
- **Clear project structure** – oddelenie dát do `data/` a grafov
  do `charts/`.
- **requirements.txt** – zoznam externých Python knižníc potrebných
  na spustenie projektu.

Ostatné voliteľné bonusy, ako Streamlit aplikácia, SQLite databáza,
logging alebo change detection, som neimplementoval.

## AI Usage & Verification

Pri projekte som používal ChatGPT ako pomoc pri programovaní a hlavne
pri zisťovaní, ako webová stránka načítava údaje o bytoch.

Najväčší problém bol, že som pôvodne hľadal funkciu `showFlat()` v
JavaScripte. Tá sa však v `map.js` nenachádzala. Po preskúmaní stránky
som zistil, že údaje o bytoch sú dostupné priamo v `flats.json`, čo
umožnilo vytvoriť scraper jednoduchšie.

AI mi pomohla najmä s:
- pochopením štruktúry stránky,
- získaním dát z JSON,
- úpravou Python kódu,
- čistením dát,
- validáciou,
- vytvorením základných grafov.

Výsledky som manuálne overil porovnaním s dátami zo stránky a kontrolou
počtu záznamov, cien, výmer a ID bytov.

AI sa na začiatku zamerala na hľadanie funkcie `showFlat()` v
JavaScripte. Táto funkcia však nebola priamo v `map.js`, preto som
musel overiť, odkiaľ sa dáta skutočne načítavajú. Nakoniec sa ukázalo,
že dostupné údaje sú priamo v `flats.json`.

Navrhnuté riešenia som upravil podľa štruktúry môjho projektu a ponechal
som iba funkcie potrebné pre zadanie. Napríklad som nepoužil Streamlit
ani change detection, keďže išlo o voliteľné bonusy.

Príklady použitých promptov:

1. „Ako zistiť, odkiaľ webová stránka načítava údaje o bytoch?“
2. „Ako môžem použiť flats.json na vytvorenie pandas DataFrame?“
3. „Môžeš skontrolovať môj scraper a navrhnúť jednoduchšie riešenie?“

## Spustenie

Na spustenie projektu je potrebné mať nainštalovaný Python
a knižnice uvedené v `requirements.txt`.

Použité externé knižnice:

- pandas
- requests
- matplotlib

Jednotlivé Python skripty je možné spustiť samostatne podľa
príslušnej časti projektu.