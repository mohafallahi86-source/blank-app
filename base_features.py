import numpy as np
import pandas as pd

# Column 289 - Quote der flüssigen Mittel (%).1
if set(['Liquide Mittel', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['quote_fluessige_mittel'] = (df['Liquide Mittel'] / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 293 - Kundenziel (Tage).1 (Days Sales Outstanding)
if set(['Forderungen aus Lieferungen und Leistungen RLZ bis 1 Jahr', 'Umsatzerlöse']).issubset(df.columns):
    # Note: Has scaling issues - for large companies (Bilanzsumme > 100), multiply Umsatz by 1000
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['kundenziel_tage'] = (df['Forderungen aus Lieferungen und Leistungen RLZ bis 1 Jahr'] / (df['Umsatzerlöse'] * scale_factor)) * 365

# Column 297 - Eigenkapitalquote (%).1 (Equity Ratio)
if set(['bereinigtes Eigenkapital', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['eigenkapitalquote'] = (df['bereinigtes Eigenkapital'] / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 301 - Verschuldungsgrad.1 (Debt-to-Equity Ratio)
if set(['Summe Verbindlichkeiten', 'bereinigtes Eigenkapital']).issubset(df.columns):
    df['verschuldungsgrad'] = (df['Summe Verbindlichkeiten'] / df['bereinigtes Eigenkapital']) * 100

# Column 305 - Kurzfristige Fremdkapitalquote (%).1
if set(['kurzfristiges Fremdkapital', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['kurzfr_fk_quote'] = (df['kurzfristiges Fremdkapital'] / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 309 - Langfristige Fremdkapitalquote (%).1
if set(['langfristiges Fremdkapital', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['langfr_fk_quote'] = (df['langfristiges Fremdkapital'] / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 313 - kurzfristige Kapitalbindung (%).1
if set(['Umlaufvermögen', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['kurzfr_kapitalbindung'] = (df['Umlaufvermögen'] / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 317 - Kapitalbindungsdauer (Tage).1
if set(['Umlaufvermögen', 'Umsatzerlöse']).issubset(df.columns):
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['kapitalbindungsdauer'] = (df['Umlaufvermögen'] / (df['Umsatzerlöse'] * scale_factor)) * 365

# Column 321 - Fremdkapitalstruktur (%).1
if set(['kurzfristiges Fremdkapital', 'Summe Verbindlichkeiten']).issubset(df.columns):
    df['fk_struktur'] = (df['kurzfristiges Fremdkapital'] / df['Summe Verbindlichkeiten']) * 100

# Column 325 - Quote Verbindl. aus Lieferungen und Leistungen mod.(%).1
if set(['Verbindlichkeiten aus Lieferungen und Leistungen RLZ bis 1 Jahr', 'kurzfristiges Fremdkapital']).issubset(df.columns):
    df['lieferanten_verb_quote'] = (df['Verbindlichkeiten aus Lieferungen und Leistungen RLZ bis 1 Jahr'] / df['kurzfristiges Fremdkapital']) * 100

# Column 329 - Lieferantenziel (Tage).1 (Days Payable Outstanding)
if set(['Verbindlichkeiten aus Lieferungen und Leistungen RLZ bis 1 Jahr', 'Aufwand für Roh-, Hilfs- und Betriebsstoffe', 'Aufwand für bezogene Leistungen']).issubset(df.columns):
    material_costs = df['Aufwand für Roh-, Hilfs- und Betriebsstoffe'] + df['Aufwand für bezogene Leistungen']
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['lieferantenziel'] = (df['Verbindlichkeiten aus Lieferungen und Leistungen RLZ bis 1 Jahr'] / (material_costs * scale_factor)) * 365

# Column 333 - Cash Flow (absolut).1
if set(['Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung']).issubset(df.columns):
    df['cash_flow'] = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']

# Column 337 - Cash Flow zur Gesamtleistung (%).1
if set(['Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung', 'Gesamtleistung']).issubset(df.columns):
    cash_flow = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']
    df['cf_zu_gesamtleistung'] = (cash_flow / df['Gesamtleistung']) * 100

# Column 341 - Cash Flow zur Effektivverschuldung (%).1
if set(['Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung', 'Summe Verbindlichkeiten', 'Liquide Mittel']).issubset(df.columns):
    cash_flow = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']
    effektiv_verschuldung = df['Summe Verbindlichkeiten'] - df['Liquide Mittel']
    df['cf_zu_effektivverschuldung'] = (cash_flow / effektiv_verschuldung) * 100

# Column 345 - Cash Flow ROI (%).1
if set(['Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    cash_flow = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']
    df['cash_flow_roi'] = (cash_flow / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 349 - Dynamische Entschuldungsdauer (Jahre).1
if set(['Summe Verbindlichkeiten', 'Liquide Mittel', 'Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung']).issubset(df.columns):
    effektiv_verschuldung = df['Summe Verbindlichkeiten'] - df['Liquide Mittel']
    cash_flow = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']
    df['dynamische_entschuldungsdauer'] = effektiv_verschuldung / cash_flow
    df['dynamische_entschuldungsdauer'] = df['dynamische_entschuldungsdauer'].replace([np.inf, -np.inf], np.nan)

# Column 353 - Schuldendienstfähigkeit (%).1
if set(['Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung', 'Finanzergebnis']).issubset(df.columns):
    cash_flow = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']
    df['schuldendienstfaehigkeit'] = (cash_flow / abs(df['Finanzergebnis'])) * 100
    df['schuldendienstfaehigkeit'] = df['schuldendienstfaehigkeit'].replace([np.inf, -np.inf], np.nan)

# Column 357 - Return on Investment (%).1
if set(['Betriebsergebnis', 'Finanzergebnis', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['roi'] = ((df['Betriebsergebnis'] + df['Finanzergebnis']) / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 361 - Eigenkapitalrentabilität (%).1
if set(['Jahresergebnis', 'bereinigtes Eigenkapital']).issubset(df.columns):
    df['eigenkapitalrentabilitaet'] = (df['Jahresergebnis'] / df['bereinigtes Eigenkapital']) * 100
    df['eigenkapitalrentabilitaet'] = df['eigenkapitalrentabilitaet'].replace([np.inf, -np.inf], np.nan)

# Column 365 - Gesamtkapitalrentabilität (%).1
if set(['Betriebsergebnis', 'Finanzergebnis', 'bereinigte Bilanzsumme Aktiva']).issubset(df.columns):
    df['gesamtkapitalrentabilitaet'] = ((df['Betriebsergebnis'] + abs(df['Finanzergebnis'])) / df['bereinigte Bilanzsumme Aktiva']) * 100

# Column 369 - Umsatzrentabilität (%).1 (Profit Margin)
if set(['Betriebsergebnis', 'Umsatzerlöse']).issubset(df.columns):
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['umsatzrentabilitaet'] = (df['Betriebsergebnis'] / (df['Umsatzerlöse'] * scale_factor)) * 100

# Column 373 - Rohertragsquote (%).1 (Gross Margin)
if set(['Rohertrag', 'Gesamtleistung']).issubset(df.columns):
    df['rohertragsquote'] = (df['Rohertrag'] / df['Gesamtleistung']) * 100

# Column 377 - EBIT zum Zinsaufwand.1 (Interest Coverage Ratio)
if set(['Betriebsergebnis', 'Finanzergebnis']).issubset(df.columns):
    df['ebit_zu_zinsaufwand'] = df['Betriebsergebnis'] / abs(df['Finanzergebnis'])
    df['ebit_zu_zinsaufwand'] = df['ebit_zu_zinsaufwand'].replace([np.inf, -np.inf], np.nan)

# Column 381 - EBITDA zum Zinsaufwand.1
if set(['Betriebsergebnis', 'Abschreibungen inkl. Firmenabschreibung', 'Finanzergebnis']).issubset(df.columns):
    ebitda = df['Betriebsergebnis'] + df['Abschreibungen inkl. Firmenabschreibung']
    df['ebitda_zu_zinsaufwand'] = ebitda / abs(df['Finanzergebnis'])
    df['ebitda_zu_zinsaufwand'] = df['ebitda_zu_zinsaufwand'].replace([np.inf, -np.inf], np.nan)

# Column 385 - Personalaufwandsquote (%).1
if set(['Löhne und Gehälter', 'soziale Abgaben, Altersversorgung', 'Gesamtleistung']).issubset(df.columns):
    personalaufwand = df['Löhne und Gehälter'] + df['soziale Abgaben, Altersversorgung']
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['personalaufwandsquote'] = (personalaufwand / (df['Gesamtleistung'] * scale_factor)) * 100

# Column 389 - Materialaufwandsquote (%).1
if set(['Aufwand für Roh-, Hilfs- und Betriebsstoffe', 'Aufwand für bezogene Leistungen', 'Gesamtleistung']).issubset(df.columns):
    materialaufwand = df['Aufwand für Roh-, Hilfs- und Betriebsstoffe'] + df['Aufwand für bezogene Leistungen']
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['materialaufwandsquote'] = (materialaufwand / (df['Gesamtleistung'] * scale_factor)) * 100

# Column 393 - Aufwand-Ertrag-Verhältnis.1
if set(['Gesamtleistung', 'sonstige betriebliche Erträge', 'Betriebsergebnis']).issubset(df.columns):
    gesamtertrag = df['Gesamtleistung'] + df['sonstige betriebliche Erträge']
    gesamtaufwand = gesamtertrag - df['Betriebsergebnis']
    df['aufwand_ertrag_verhaeltnis'] = gesamtaufwand / gesamtertrag
    df['aufwand_ertrag_verhaeltnis'] = df['aufwand_ertrag_verhaeltnis'].replace([np.inf, -np.inf], np.nan)

# Column 397 - Umsatz je Mitarbeiter (absolut).1
if 'Mitarbeiteranzahl' in df.columns and 'Umsatzerlöse' in df.columns:
    scale_factor = np.where(df['bereinigte Bilanzsumme Aktiva'] > 100, 1000, 1)
    df['umsatz_je_mitarbeiter'] = (df['Umsatzerlöse'] * scale_factor) / df['Mitarbeiteranzahl']
    df['umsatz_je_mitarbeiter'] = df['umsatz_je_mitarbeiter'].replace([np.inf, -np.inf], np.nan)

# Column 401 - Zinsaufwand zum Fremdkapital (%).1
if set(['Finanzergebnis', 'Summe Verbindlichkeiten']).issubset(df.columns):
    df['zinsaufwand_zu_fk'] = (abs(df['Finanzergebnis']) / df['Summe Verbindlichkeiten']) * 100

# Column 405 - Erfolgsquote (%).1
if set(['Jahresergebnis', 'Gesamtleistung']).issubset(df.columns):
    df['erfolgsquote'] = (df['Jahresergebnis'] / df['Gesamtleistung']) * 100

# Column 409 - Liquidität I. Grades (%).1 (Cash Ratio)
if set(['Liquide Mittel', 'kurzfristiges Fremdkapital']).issubset(df.columns):
    df['liquiditaet_1'] = (df['Liquide Mittel'] / df['kurzfristiges Fremdkapital']) * 100
    df['liquiditaet_1'] = df['liquiditaet_1'].replace([np.inf, -np.inf], np.nan)

# Column 413 - Liquidität II. Grades (%).1 (Quick Ratio)
if set(['Liquide Mittel', 'Forderungen aus Lieferungen und Leistungen RLZ bis 1 Jahr', 'kurzfristiges Fremdkapital']).issubset(df.columns):
    df['liquiditaet_2'] = ((df['Liquide Mittel'] + df['Forderungen aus Lieferungen und Leistungen RLZ bis 1 Jahr']) / df['kurzfristiges Fremdkapital']) * 100
    df['liquiditaet_2'] = df['liquiditaet_2'].replace([np.inf, -np.inf], np.nan)

# Column 417 - Liquidität III. Grades (%).1 (Current Ratio)
if set(['Umlaufvermögen', 'kurzfristiges Fremdkapital']).issubset(df.columns):
    df['liquiditaet_3'] = (df['Umlaufvermögen'] / df['kurzfristiges Fremdkapital']) * 100
    df['liquiditaet_3'] = df['liquiditaet_3'].replace([np.inf, -np.inf], np.nan)

# Column 421 - Net Working Capital (absolut).1
if set(['Umlaufvermögen', 'kurzfristiges Fremdkapital']).issubset(df.columns):
    df['net_working_capital'] = df['Umlaufvermögen'] - df['kurzfristiges Fremdkapital']

# Column 425 - Liquidität I. Grades (%) erweitert.1
if set(['Liquide Mittel', 'Wertpapiere des Umlaufvermögens', 'kurzfristiges Fremdkapital']).issubset(df.columns):
    df['liquiditaet_1_erweitert'] = ((df['Liquide Mittel'] + df['Wertpapiere des Umlaufvermögens']) / df['kurzfristiges Fremdkapital']) * 100
    df['liquiditaet_1_erweitert'] = df['liquiditaet_1_erweitert'].replace([np.inf, -np.inf], np.nan)
