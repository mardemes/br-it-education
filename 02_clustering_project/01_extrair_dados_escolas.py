"""
PROJETO: Cluster Analysis - Perfis Digitais das Escolas Brasileiras
FASE 1: Extração e Preparação de Dados - TIC Educação 2024 (Escolas)

Este script:
1. Lê o arquivo Excel com dados TIC Educação 2024 - Escolas
2. Extrai as sheets mais relevantes para cluster analysis
3. Transforma em formato estruturado (DataFrame)
4. Salva em CSV e JSON para análise posterior

Autor: [Seu nome]
Data: 2025
"""

import pandas as pd
import numpy as np
import openpyxl
from pathlib import Path
import json

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Caminho do arquivo (ajuste conforme necessário)
ARQUIVO_EXCEL = '01_analises/tic_educacao_2024_escolas_tabela_total_v1.0.xlsx'
PASTA_OUTPUT = 'dados_processados'

# Sheets prioritárias para extração
SHEETS_PRIORITARIAS = {
    'infraestrutura': ['A1', 'B1', 'B1C', 'B2'],
    'conectividade': ['A2', 'A3_1', 'A4', 'C1'],
    'gestao_uso': ['E1', 'E1A', 'F1', 'F2', 'G4'],
    'contexto': ['K3', 'K6']
}

# Todas as sheets em lista plana
TODAS_SHEETS = [s for grupo in SHEETS_PRIORITARIAS.values() for s in grupo]

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def criar_pasta_output():
    """Cria pasta para arquivos de saída se não existir"""
    Path(PASTA_OUTPUT).mkdir(exist_ok=True)
    print(f"✓ Pasta '{PASTA_OUTPUT}' criada/verificada")


def ler_sheet_estruturada(arquivo, sheet_name, verbose=True):
    """
    Lê uma sheet do Excel e retorna DataFrame estruturado.
    
    Estrutura esperada das sheets TIC:
    - Linha 0: Título (ex: "A1 - ESCOLAS COM ACESSO À INTERNET")
    - Linha 1: Descrição
    - Linha 2-3: Headers
    - Linha 4+: Dados
    """
    df_raw = pd.read_excel(arquivo, sheet_name=sheet_name, header=None)
    
    # Extrair metadados
    titulo = df_raw.iloc[0, 0] if not pd.isna(df_raw.iloc[0, 0]) else sheet_name
    
    # Headers geralmente nas linhas 2-3
    header_row1 = df_raw.iloc[2].fillna('')
    header_row2 = df_raw.iloc[3].fillna('')
    
    # Combinar headers (quando há sub-colunas)
    headers = []
    for i, (h1, h2) in enumerate(zip(header_row1, header_row2)):
        if h1 and h2 and h1 != h2:
            headers.append(f"{h1}_{h2}")
        elif h1:
            headers.append(str(h1))
        elif h2:
            headers.append(str(h2))
        else:
            headers.append(f"col_{i}")
    
    # Criar DataFrame com dados (linha 4 em diante)
    df = df_raw.iloc[4:].copy()
    df.columns = headers[:len(df.columns)]
    
    # Limpar linhas vazias
    df = df.dropna(how='all')
    
    # Resetar índice
    df = df.reset_index(drop=True)
    
    if verbose:
        print(f"  ✓ {sheet_name}: {len(df)} linhas, {len(df.columns)} colunas")
    
    return df, titulo


def extrair_categorias_e_valores(df):
    """
    Extrai estrutura hierárquica das categorias (REGIÃO, ÁREA, etc.)
    e seus valores correspondentes.
    
    Returns:
        DataFrame pivotado onde cada linha é uma observação única
    """
    # Identificar colunas de categoria vs valores
    primeira_col = df.columns[0]
    segunda_col = df.columns[1] if len(df.columns) > 1 else None
    
    # Criar identificador único para cada observação
    df['categoria_principal'] = df[primeira_col]
    df['categoria_secundaria'] = df[segunda_col] if segunda_col else ''
    
    # Criar ID único
    df['observacao_id'] = df['categoria_principal'] + '_' + df['categoria_secundaria'].fillna('Total')
    df['observacao_id'] = df['observacao_id'].str.replace(' ', '_')
    
    return df


def processar_sheet(arquivo, sheet_name):
    """
    Processa uma sheet completa e retorna DataFrame limpo
    """
    df, titulo = ler_sheet_estruturada(arquivo, sheet_name, verbose=False)
    df = extrair_categorias_e_valores(df)
    
    # Adicionar metadados
    df['sheet_origem'] = sheet_name
    df['titulo_sheet'] = titulo
    
    return df


def extrair_features_numericas(df):
    """
    Extrai apenas colunas numéricas relevantes (features)
    Remove colunas de categoria e metadados
    """
    # Colunas a ignorar
    colunas_ignorar = ['categoria_principal', 'categoria_secundaria', 
                       'observacao_id', 'sheet_origem', 'titulo_sheet']
    
    # Selecionar colunas numéricas
    colunas_numericas = []
    for col in df.columns:
        if col not in colunas_ignorar:
            # Tentar converter para numérico
            try:
                pd.to_numeric(df[col], errors='coerce')
                colunas_numericas.append(col)
            except:
                pass
    
    return df[colunas_ignorar + colunas_numericas]


# ============================================================================
# FUNÇÃO PRINCIPAL DE EXTRAÇÃO
# ============================================================================

def extrair_dados_todas_sheets(arquivo=ARQUIVO_EXCEL, sheets=TODAS_SHEETS):
    """
    Extrai dados de todas as sheets prioritárias e consolida
    """
    print("\n" + "="*70)
    print("INICIANDO EXTRAÇÃO DE DADOS - TIC EDUCAÇÃO 2024 (ESCOLAS)")
    print("="*70 + "\n")
    
    # Verificar se arquivo existe
    if not Path(arquivo).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")
    
    print(f"✓ Arquivo encontrado: {arquivo}\n")
    
    # Dicionário para armazenar DataFrames
    dfs_por_sheet = {}
    
    print("Extraindo sheets...")
    for i, sheet in enumerate(sheets, 1):
        try:
            df = processar_sheet(arquivo, sheet)
            dfs_por_sheet[sheet] = df
            print(f"  [{i}/{len(sheets)}] ✓ {sheet}: {len(df)} observações")
        except Exception as e:
            print(f"  [{i}/{len(sheets)}] ✗ {sheet}: ERRO - {str(e)}")
    
    print(f"\n✓ Total de sheets extraídas: {len(dfs_por_sheet)}/{len(sheets)}")
    
    return dfs_por_sheet


def criar_dataset_consolidado(dfs_por_sheet):
    """
    Cria dataset consolidado com todas as features
    Cada linha = uma observação (ex: "REGIÃO_Norte", "ÁREA_Urbana")
    Cada coluna = uma feature de uma sheet específica
    """
    print("\n" + "-"*70)
    print("CONSOLIDANDO DADOS...")
    print("-"*70 + "\n")
    
    # Identificar todas as observações únicas
    todas_obs = set()
    for df in dfs_por_sheet.values():
        todas_obs.update(df['observacao_id'].unique())
    
    print(f"Total de observações únicas encontradas: {len(todas_obs)}")
    
    # Criar DataFrame base com todas as observações
    df_consolidado = pd.DataFrame({'observacao_id': sorted(todas_obs)})
    
    # Para cada sheet, adicionar suas features
    for sheet_name, df in dfs_por_sheet.items():
        # Selecionar apenas colunas numéricas
        df_num = extrair_features_numericas(df)
        
        # Renomear colunas para incluir origem
        feature_cols = [c for c in df_num.columns if c not in 
                       ['categoria_principal', 'categoria_secundaria', 
                        'observacao_id', 'sheet_origem', 'titulo_sheet']]
        
        rename_dict = {col: f"{sheet_name}_{col}" for col in feature_cols}
        df_num = df_num.rename(columns=rename_dict)
        
        # Merge com dataset consolidado
        df_consolidado = df_consolidado.merge(
            df_num[['observacao_id'] + list(rename_dict.values())],
            on='observacao_id',
            how='left'
        )
    
    print(f"✓ Dataset consolidado: {len(df_consolidado)} observações × {len(df_consolidado.columns)-1} features")
    
    return df_consolidado


def salvar_resultados(df_consolidado, dfs_por_sheet):
    """
    Salva resultados em múltiplos formatos
    """
    print("\n" + "-"*70)
    print("SALVANDO RESULTADOS...")
    print("-"*70 + "\n")
    
    criar_pasta_output()
    
    # 1. Dataset consolidado em CSV
    arquivo_csv = f"{PASTA_OUTPUT}/escolas_2024_consolidado.csv"
    df_consolidado.to_csv(arquivo_csv, index=False, encoding='utf-8-sig')
    print(f"✓ CSV salvo: {arquivo_csv}")
    
    # 2. Dataset consolidado em JSON
    arquivo_json = f"{PASTA_OUTPUT}/escolas_2024_consolidado.json"
    df_consolidado.to_json(arquivo_json, orient='records', force_ascii=False, indent=2)
    print(f"✓ JSON salvo: {arquivo_json}")
    
    # 3. Sheets individuais em CSV (para referência)
    pasta_sheets = f"{PASTA_OUTPUT}/sheets_individuais"
    Path(pasta_sheets).mkdir(exist_ok=True)
    
    for sheet_name, df in dfs_por_sheet.items():
        arquivo = f"{pasta_sheets}/{sheet_name}.csv"
        df.to_csv(arquivo, index=False, encoding='utf-8-sig')
    
    print(f"✓ Sheets individuais salvas em: {pasta_sheets}/")
    
    # 4. Metadados
    metadados = {
        'total_observacoes': len(df_consolidado),
        'total_features': len(df_consolidado.columns) - 1,
        'sheets_extraidas': list(dfs_por_sheet.keys()),
        'observacoes_unicas': df_consolidado['observacao_id'].tolist(),
        'colunas': df_consolidado.columns.tolist()
    }
    
    arquivo_meta = f"{PASTA_OUTPUT}/metadados.json"
    with open(arquivo_meta, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Metadados salvos: {arquivo_meta}")
    
    # 5. Relatório de extração
    relatorio = f"""
RELATÓRIO DE EXTRAÇÃO - TIC EDUCAÇÃO 2024 (ESCOLAS)
{'='*70}

RESUMO:
- Total de observações: {len(df_consolidado)}
- Total de features: {len(df_consolidado.columns) - 1}
- Sheets processadas: {len(dfs_por_sheet)}

OBSERVAÇÕES EXTRAÍDAS:
{chr(10).join(['  - ' + obs for obs in sorted(df_consolidado['observacao_id'].unique())[:20]])}
  ... (total: {len(df_consolidado)})

FEATURES POR CATEGORIA:
"""
    
    for categoria, sheets in SHEETS_PRIORITARIAS.items():
        relatorio += f"\n{categoria.upper()}:\n"
        for sheet in sheets:
            if sheet in dfs_por_sheet:
                cols = [c for c in df_consolidado.columns if c.startswith(sheet + '_')]
                relatorio += f"  - {sheet}: {len(cols)} features\n"
    
    arquivo_relatorio = f"{PASTA_OUTPUT}/relatorio_extracao.txt"
    with open(arquivo_relatorio, 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    print(f"✓ Relatório salvo: {arquivo_relatorio}")


def gerar_estatisticas_descritivas(df):
    """
    Gera estatísticas descritivas do dataset
    """
    print("\n" + "="*70)
    print("ESTATÍSTICAS DESCRITIVAS")
    print("="*70 + "\n")
    
    # Info básica
    print(f"Shape: {df.shape}")
    print(f"Observações: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")  # -1 para observacao_id
    
    # Valores faltantes
    print("\n--- Valores Faltantes ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Total': missing,
        'Percentual (%)': missing_pct
    })
    missing_df = missing_df[missing_df['Total'] > 0].sort_values('Total', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\nColunas com valores faltantes: {len(missing_df)}")
        print(missing_df.head(10))
    else:
        print("✓ Nenhum valor faltante!")
    
    # Estatísticas das features numéricas
    print("\n--- Estatísticas das Features Numéricas (amostra) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().iloc[:, :5])  # Primeiras 5 colunas


# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Função principal - executa todo o pipeline
    """
    try:
        # 1. Extrair dados de todas as sheets
        dfs_por_sheet = extrair_dados_todas_sheets()
        
        # 2. Consolidar em um único dataset
        df_consolidado = criar_dataset_consolidado(dfs_por_sheet)
        
        # 3. Salvar resultados
        salvar_resultados(df_consolidado, dfs_por_sheet)
        
        # 4. Estatísticas descritivas
        gerar_estatisticas_descritivas(df_consolidado)
        
        print("\n" + "="*70)
        print("✓ EXTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*70 + "\n")
        
        print("Próximos passos:")
        print("  1. Revisar o arquivo: dados_processados/escolas_2024_consolidado.csv")
        print("  2. Analisar estatísticas descritivas")
        print("  3. Limpar/tratar valores faltantes")
        print("  4. Normalizar features")
        print("  5. Executar cluster analysis!")
        
        return df_consolidado
        
    except Exception as e:
        print(f"\n✗ ERRO durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    df_resultado = main()