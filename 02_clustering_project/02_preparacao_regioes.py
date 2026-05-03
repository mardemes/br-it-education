import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("FASE 1: PREPARAÇÃO DE DADOS - CLUSTERING POR REGIÃO")
print("="*70)

# 1. CARREGAR DADOS
df = pd.read_csv('01_analises/escolas_2024_consolidado.csv')
print(f"\n✓ Dataset carregado: {df.shape}")

# 2. FILTRAR APENAS REGIÕES
df_regioes = df[df['observacao_id'].str.contains('REGIÃO_', na=False)].copy()
df_regioes = df_regioes[~df_regioes['observacao_id'].str.contains('TOTAL')]

print(f"✓ Regiões filtradas: {len(df_regioes)}")
print("\nRegiões encontradas:")
for idx, regiao in enumerate(df_regioes['observacao_id'].values, 1):
    print(f"  {idx}. {regiao}")

# 3. ANÁLISE DE VALORES FALTANTES
print("\n" + "-"*70)
print("ANÁLISE DE VALORES FALTANTES")
print("-"*70)

missing_by_col = df_regioes.isnull().sum()
missing_pct = (missing_by_col / len(df_regioes) * 100)

# Features com dados completos
complete_features = missing_pct[missing_pct == 0].index.tolist()
complete_features = [f for f in complete_features if f != 'observacao_id']

print(f"\nFeatures com dados completos: {len(complete_features)}")
print(f"Features com algum NaN: {len(missing_pct[missing_pct > 0])}")

# 4. SELECIONAR APENAS FEATURES COMPLETAS
df_clean = df_regioes[['observacao_id'] + complete_features].copy()

print(f"\n✓ Dataset limpo: {df_clean.shape}")
print(f"  - Observações: {len(df_clean)}")
print(f"  - Features: {len(complete_features)}")

# 5. SEPARAR FEATURES NUMÉRICAS
X = df_clean.drop('observacao_id', axis=1)
regioes_nomes = df_clean['observacao_id'].values

# Converter tudo para numérico (lidar com strings)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Remover colunas que ficaram todas NaN
X = X.dropna(axis=1, how='all')

print(f"\n✓ Features numéricas: {X.shape[1]}")

# 6. VERIFICAR VARIÂNCIA
print("\n" + "-"*70)
print("ANÁLISE DE VARIÂNCIA")
print("-"*70)

variances = X.var()
zero_var = variances[variances == 0].index.tolist()

if len(zero_var) > 0:
    print(f"\n⚠️  Features com variância zero (constantes): {len(zero_var)}")
    print("Removendo features constantes...")
    X = X.drop(columns=zero_var)
else:
    print("\n✓ Nenhuma feature constante encontrada")

print(f"\n✓ Features finais: {X.shape[1]}")

# 7. NORMALIZAÇÃO
print("\n" + "-"*70)
print("NORMALIZAÇÃO")
print("-"*70)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)

print("✓ Dados normalizados (média=0, std=1)")

# 8. SALVAR DADOS PREPARADOS
df_preparado = X_normalized.copy()
df_preparado.insert(0, 'observacao_id', regioes_nomes)

output_file = 'dados_processados/regioes_preparado_para_clustering.csv'
df_preparado.to_csv(output_file, index=False)

print(f"\n✓ Dados salvos: {output_file}")

# 9. ESTATÍSTICAS FINAIS
print("\n" + "="*70)
print("RESUMO DOS DADOS PREPARADOS")
print("="*70)

print(f"\nObservações: {len(df_preparado)}")
print(f"Features: {len(X.columns)}")
print(f"\nRegiões:")
for regiao in regioes_nomes:
    print(f"  - {regiao}")

print(f"\nPrimeiras 10 features:")
for i, col in enumerate(X.columns[:10], 1):
    print(f"  {i}. {col}")

print("\n" + "="*70)
print("✓ DADOS PRONTOS PARA CLUSTERING!")
print("="*70)
print("\nPróximo passo: Executar cluster analysis")
print("  Script: 03_clustering_regioes.py")