import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import uuid

expected_features = [
    'ACKR3', 'ALB', 'APCDD1', 'APOA1', 'APOH', 'BHMT', 'C3', 'CCKBR', 'CCL2', 'CDH11',
    'CDX1', 'CER1', 'COLEC12', 'CXCR4', 'CYP26A1', 'CYP2E1', 'CYP4A11', 'DACT1', 'DEFA5',
    'DKK1', 'DLK1', 'EDNRB', 'EGFLAM', 'EOMES', 'EPHA4', 'ERBB4', 'FEZF1', 'FGB', 'FGF17',
    'FGF8', 'FZD10', 'GAD1', 'GATA6', 'GC', 'GFI1', 'GNMT', 'GREB1L', 'GRPR', 'GSC',
    'H2AC19', 'H4C15', 'HAS2', 'HMGCS2', 'HOXB3', 'HOXB9', 'HPD', 'HPR', 'HPX', 'HSPA1B',
    'IDO1', 'IGFBP5', 'LEF1', 'LEFTY1', 'LEFTY2', 'LRIG3', 'MAT1A', 'MGST2', 'MIXL1', 'MLLT3',
    'MSGN1', 'MSX1', 'MSX2', 'MT1M', 'MUCL1', 'NCAM1', 'NKD1', 'NNAT', 'NOTUM', 'NPPB',
    'NXPH4', 'PAX6', 'PCK1', 'PLSCR2', 'PRSS23', 'PRTG', 'RBP1', 'RBP4', 'RHOBTB3', 'RNASE1',
    'RPS4Y1', 'S1PR3', 'SERPINA1', 'SHISA2', 'SHISAL2B', 'SOX17', 'SP5', 'SPP1', 'TF',
    'TNFRSF11B', 'TTR', 'URAD', 'VTN', 'WNT5B', 'ZEB2', 'ZFHX4', 'ZNF521'
]


def load_expression_table(file_path):
    """
    Load gene expression matrix from .csv, .tsv, .xlsx, or .txt.
    Assumes:
    - Gene names in first column
    - First row contains sample names
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.csv':
            df = pd.read_csv(file_path, header=0, index_col=0)
        elif ext == '.tsv':
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=0)
        elif ext == '.xlsx':
            df = pd.read_excel(file_path, header=0, index_col=0)
        elif ext == '.txt':
            # Try auto-detect for .txt (it could be comma, tab, or space)
            df = pd.read_csv(file_path, sep=None, engine='python', header=0, index_col=0)
        else:
            raise ValueError("Unsupported file type. Please upload .csv, .tsv, .txt, or .xlsx.")
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    # Validate shape
    if df.shape[1] == 0:
        raise ValueError("No sample columns detected. Make sure the first row contains sample names and the first column contains gene names.")
    if df.shape[0] == 0:
        raise ValueError("No genes detected. Make sure gene names are in the first column.")

    # Force numeric just in case
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


def validate_prepare_data(input_data):
    data = input_data.copy()

    if data.shape[1] == 0:
        raise ValueError("No sample columns detected.")

    # Track how many features are missing
    missing_features = [f for f in expected_features if f not in data.index]

    # Fill missing ones with zeros
    for feature in missing_features:
        data.loc[feature] = [0] * data.shape[1]

    # Strictly reorder to match model input
    try:
        data = data.loc[expected_features].T
    except KeyError as e:
        raise ValueError(f"Failed to align features for model input: {e}")

    return data


def predict_and_score(model, data, label_mapping, class_to_score):
    predicted_indices = model.predict(data)
    probabilities = model.predict_proba(data)
    predicted_classes = [label_mapping[i] for i in predicted_indices]

    weighted_scores = np.sum([
        probabilities[:, i] * class_to_score[label_mapping[class_idx]]
        for i, class_idx in enumerate(model.classes_)
    ], axis=0)

    confidence = np.max(probabilities, axis=1)  # max probability per row as confidence

    return weighted_scores, confidence

def predict_ipsc_potential(input_data,
                           endo_model_path,
                           meso_model_path,
                           ecto_model_path):
    endo_model = joblib.load(endo_model_path)
    meso_model = joblib.load(meso_model_path)
    ecto_model = joblib.load(ecto_model_path)

    data_prepared = validate_prepare_data(input_data)

    # Predict with all three models
    endoscore, endo_conf = predict_and_score(
        endo_model, data_prepared,
        {0:'bad',1:'excellent',2:'fair',3:'normal',4:'poor'},
        {'bad':1,'poor':2,'fair':3,'normal':4,'excellent':5}
    )

    mesoscore, meso_conf = predict_and_score(
        meso_model, data_prepared,
        {0:'good',1:'higher_normal',2:'lower_normal',3:'normal'},
        {'lower_normal':2,'normal':3,'higher_normal':4,'good':5}
    )

    ectoscore, ecto_conf = predict_and_score(
        ecto_model, data_prepared,
        {0:'good',1:'higher_normal',2:'lower_normal',3:'normal'},
        {'lower_normal':2,'normal':3,'higher_normal':4,'good':5}
    )

    # Combine scores
    summary_df = pd.DataFrame({
        'endoscore': endoscore,
        'mesoscore': mesoscore,
        'ectoscore': ectoscore
    }, index=data_prepared.index)

    summary_df['pluriscore'] = summary_df[['endoscore', 'mesoscore', 'ectoscore']].mean(axis=1)
    summary_df['confidence'] = (endo_conf + meso_conf + ecto_conf) / 3

    return summary_df

def run_models(file_path):
    df = load_expression_table(file_path)
    summary_df = predict_ipsc_potential(
        df,
        endo_model_path='models/endo_model.pkl',
        meso_model_path='models/meso_model.pkl',
        ecto_model_path='models/ecto_model.pkl'
    )

    result_dict = summary_df.to_dict(orient='index')

    if summary_df.shape[0] == 1:
        bar_img, bar_file = plot_single_sample_barplot(summary_df.iloc[0])
        jitter_img, jitter_file = None, None
    else:
        bar_img, bar_file = None, None
        jitter_img, jitter_file = plot_jittered_scores(summary_df)

    return result_dict, bar_img, jitter_img, bar_file, jitter_file


def plot_single_sample_barplot(row):
    plt.figure(figsize=(6, 4))
    row = row[['endoscore', 'mesoscore', 'ectoscore', 'pluriscore']]
    row.plot(kind='barh', color='skyblue')
    plt.title("PluriScore")
    plt.xlim(0, 5)
    plt.tight_layout()

    filename = f"{uuid.uuid4().hex}_barplot.png"
    path = os.path.join('static/plots', filename)
    base64_img = _plot_to_base64(path)
    return base64_img, filename

def plot_jittered_scores(summary_df):
    df_melted = summary_df[['endoscore', 'mesoscore', 'ectoscore', 'pluriscore']].reset_index().melt(id_vars='index')
    df_melted.columns = ['Sample', 'ScoreType', 'Score']

    plt.figure(figsize=(10, 5))
    
    # Violin plot behind
    sns.violinplot(data=df_melted, x='ScoreType', y='Score', inner=None, alpha=0.3, palette='pastel')

    # Stripplot overlaid
    sns.stripplot(data=df_melted, x='ScoreType', y='Score', jitter=True, size=5,
                  hue='ScoreType', dodge=False, palette='Set1', legend=False)

    # Remove spines (top and right)
    sns.despine(top=True, right=True)

    # Highlight top and bottom scoring samples
    top_samples = df_melted.groupby('ScoreType')['Score'].idxmax()
    bottom_samples = df_melted.groupby('ScoreType')['Score'].idxmin()
    highlight_idx = top_samples.tolist() + bottom_samples.tolist()
    for idx in highlight_idx:
        row = df_melted.iloc[idx]
        plt.text(x=row['ScoreType'], y=row['Score'] + 0.1,
                 s=row['Sample'], fontsize=8, ha='center', color='black')

    plt.ylim(0, 5.5)
    plt.xlabel('')
    plt.ylabel('Score')
    plt.title('Score Distribution')
    plt.tight_layout()

    filename = f"{uuid.uuid4().hex}_jitterplot.png"
    path = os.path.join('static/plots', filename)
    base64_img = _plot_to_base64(path)
    return base64_img, filename


def _plot_to_base64(save_path=None):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Also save file to disk if save_path is given
    if save_path:
        plt.savefig(save_path)

    plt.close()
    return base64_img