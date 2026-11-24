from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA  # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import r2_score  # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler  # pyright: ignore[reportMissingModuleSource]


def load_health_data(filepath: Union[str, Path] = "health_study_dataset.csv") -> pd.DataFrame:
    """Läser in hälsostudiens CSV-fil som en pandas DataFrame.

    Args:
        filepath: Sökväg till CSV-filen.

    Returns:
        Inläst dataset som en pandas DataFrame.
    """
    return pd.read_csv(filepath)


def simulate_disease(df: pd.DataFrame, n_sim: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Simulerar sjukdom för n_sim personer med samma andel som i datan.

    Args:
        df: DataFrame med kolumnen ``disease``.
        n_sim: Antal simulerade individer.
        seed: Slump-seed för reproducerbarhet.

    Returns:
        En tuple med ``(verklig_andel, simulerad_andel)``.
    """
    rng = np.random.default_rng(seed)
    real_prop = float(df["disease"].mean())
    sim_data = rng.binomial(n=1, p=real_prop, size=n_sim)
    sim_prop = float(sim_data.mean())
    return real_prop, sim_prop




def test_smoking_bp(df: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Jämför systoliskt blodtryck mellan rökare och icke-rökare med t-test.

    Args:
        df: DataFrame med kolumnerna ``smoker`` och ``systolic_bp``.
        alpha: Signifikansnivå (används för tolkning i rapporten).

    Returns:
        En tuple med ``(skillnad_i_medel, t_statistika, p_varde)``.
    """
    smoker_clean = df["smoker"].astype(str).str.strip().str.lower()
    smokers = df.loc[smoker_clean.isin(["yes", "1", "true"]), "systolic_bp"].dropna()
    non_smokers = df.loc[smoker_clean.isin(["no", "0", "false"]), "systolic_bp"].dropna()
    t_stat, p_val = stats.ttest_ind(smokers, non_smokers, equal_var=False)
    diff_means = float(smokers.mean() - non_smokers.mean())
    return diff_means, float(t_stat), float(p_val)




def plot_bp_by_smoker(df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None) -> None:
    """Ritar ett lådagram över systoliskt blodtryck per rökstatus.

    Args:
        df: DataFrame med kolumnerna ``systolic_bp`` och ``smoker``.
        save_path: Valfri sökväg för att spara figuren.
    """
    plt.figure(figsize=(5, 3))
    df.boxplot(column="systolic_bp", by="smoker", grid=False)
    plt.title("Systoliskt blodtryck per rökstatus (0 = nej, 1 = ja)")
    plt.suptitle("")
    plt.xlabel("Rökare (smoker)")
    plt.ylabel("Systoliskt blodtryck (mmHg)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def disease_prevalence_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Beräknar antal, andel med sjukdom och chi-två-test per kön.

    Args:
        df: DataFrame med kolumnerna ``sex`` och ``disease``.

    Returns:
        DataFrame med antal individer, antal sjuka, andel i decimal och procent samt p-värde.

    Raises:
        ValueError: Om ``sex`` eller ``disease`` saknas i ``df``.
    """
    if "sex" not in df.columns or "disease" not in df.columns:
        raise ValueError("Kolumnerna 'sex' och 'disease' saknas i DataFrame.")

    sub = df[["sex", "disease"]].dropna().copy()
    sub["disease_num"] = pd.to_numeric(sub["disease"], errors="coerce")
    sub = sub.dropna(subset=["disease_num"])

    summary = (
        sub.groupby("sex", observed=False)["disease_num"]
        .agg(
            antal_individer="count",
            antal_sjuka="sum",
            andel_sjuka="mean",
        )
        .reset_index()
    )
    summary["andel_procent"] = (summary["andel_sjuka"] * 100).round(1)

    ct = pd.crosstab(sub["sex"], sub["disease_num"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        _, p_val, _, _ = stats.chi2_contingency(ct)
        summary["chi2_p_varde"] = round(float(p_val), 4)
    else:
        summary["chi2_p_varde"] = np.nan

    return summary


def plot_extended_analysis(df: pd.DataFrame) -> None:
    """Visualiserar sjukdomsförekomst per kön och blodtryck mot ålder med trendlinje.

    Args:
        df: DataFrame med kolumnerna ``sex``, ``disease``, ``age`` och ``systolic_bp``.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6,7))

    if "sex" in df.columns and "disease" in df.columns:
        prev_df = disease_prevalence_by_sex(df)
        axes[0].bar(
            prev_df["sex"].astype(str),
            prev_df["andel_procent"],
            color=["#3b82f6", "#f97316"][: len(prev_df)],
        )
        axes[0].set_title("Sjukdomsförekomst per kön")
        axes[0].set_xlabel("Kön")
        axes[0].set_ylabel("Andel med sjukdom (%)")
        axes[0].set_ylim(0, max(100.0, float(prev_df["andel_procent"].max()) * 1.15))
        for idx, row in prev_df.iterrows():
            axes[0].text(
                idx,
                row["andel_procent"] + 1.5,
                f"{row['andel_procent']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if "age" in df.columns and "systolic_bp" in df.columns:
        sub = df[["age", "systolic_bp"]].dropna()
        x = sub["age"].to_numpy(dtype=float)
        y = sub["systolic_bp"].to_numpy(dtype=float)

        axes[1].scatter(x, y, alpha=0.45, label="Observationer")
        if len(x) > 1:
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            axes[1].plot(
                x_line,
                y_line,
                color="red",
                linewidth=2,
                label=f"Trend (r={r_val:.2f}, p={p_val:.4f})",
            )
            axes[1].legend()

        axes[1].set_title("Blodtryck mot ålder")
        axes[1].set_xlabel("Ålder (år)")
        axes[1].set_ylabel("Systoliskt blodtryck (mmHg)")

    plt.tight_layout()
    plt.show()


def vector_matrix_bp_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Matris- och vektoranalys med numpy: normer, skalärprodukt och cosinuslikhet.

    Args:
        df: DataFrame med kolumnerna ``age``, ``weight`` och ``systolic_bp``.

    Returns:
        Ordbok med skalärprodukt, normer, cosinuslikhet och rang för ``X.T @ X``.
    """
    sub = df[["age", "weight", "systolic_bp"]].dropna()
    age_vec = sub["age"].to_numpy(dtype=float)
    weight_vec = sub["weight"].to_numpy(dtype=float)
    bp_vec = sub["systolic_bp"].to_numpy(dtype=float)

    dot_age_bp = float(np.dot(age_vec, bp_vec))
    norm_age = float(np.linalg.norm(age_vec))
    norm_bp = float(np.linalg.norm(bp_vec))
    cos_sim = float(dot_age_bp / (norm_age * norm_bp))

    X = np.column_stack([np.ones(len(sub)), age_vec, weight_vec])
    gram_rank = float(np.linalg.matrix_rank(X.T @ X))

    return {
        "dot_age_bp": dot_age_bp,
        "norm_age": norm_age,
        "norm_bp": norm_bp,
        "cos_sim_age_bp": cos_sim,
        "rank_xtx": gram_rank,
    }


def fit_bp_linear_regression(
    df: pd.DataFrame, features: Optional[List[str]] = None
) -> Tuple[LinearRegression, pd.DataFrame, float]:
    """Anpassar enkel eller multipel linjär regression för att förutsäga ``systolic_bp``.

    Args:
        df: DataFrame med funktionskolumner och ``systolic_bp``.
        features: Lista med kolumnnamn som oberoende variabler.

    Returns:
        En tuple med ``(anpassad_modell, koefficienttabell, r2_poang)``.
    """
    if features is None:
        features = ["age", "weight"]
    cols = features + ["systolic_bp"]
    sub = df[cols].dropna()

    X = sub[features].to_numpy(dtype=float)
    y = sub["systolic_bp"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = float(r2_score(y, y_pred))

    coef_table = pd.DataFrame(
        {
            "Variabel": features,
            "Koefficient": model.coef_,
        }
    )
    return model, coef_table, r2


def plot_bp_regression(df: pd.DataFrame, feature: str = "age") -> None:
    """Ritar faktisk data och linjär regressionslinje för blodtryck mot vald variabel.

    Args:
        df: DataFrame med ``feature`` och ``systolic_bp``.
        feature: Kolumnnamn på x-axeln.
    """
    sub = df[[feature, "systolic_bp"]].dropna()
    X = sub[[feature]].to_numpy(dtype=float)
    y = sub["systolic_bp"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X, y)
    x_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_grid)

    plt.figure(figsize=(5, 3))
    plt.scatter(X, y, alpha=0.5, label="Observationer")
    plt.plot(x_grid, y_line, color="red", linewidth=2, label="Linjär regression")
    plt.title(f"Blodtryck (systolic_bp) vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Systoliskt blodtryck (mmHg)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_pca_analysis(
    df: pd.DataFrame, cols: Optional[List[str]] = None, n_components: int = 2
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Utför standardiserad PCA med scikit-learn för att hitta mönster i datan.

    Args:
        df: DataFrame med numeriska variabler.
        cols: Lista med kolumner som ska ingå i PCA.
        n_components: Antal huvudkomponenter.

    Returns:
        En tuple med ``(loadings, scores, explained_variance_ratio)``.
    """
    if cols is None:
        cols = ["age", "weight", "height", "systolic_bp", "cholesterol"]
    sub = df[cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return loadings, scores, pca.explained_variance_ratio_


def plot_pca_scatter(
    df: pd.DataFrame,
    scores: np.ndarray,
    explained_var: np.ndarray,
    color_col: Optional[str] = "disease",
) -> None:
    """Ritar PC1 mot PC2 för att visualisera mönster i datan.

    Args:
        df: DataFrame som kan innehålla färgningskolumnen.
        scores: Komponentpoäng från PCA.
        explained_var: Förklarad variansandel per komponent.
        color_col: Valfri kolumn för färgkodning av punkterna.
    """
    plt.figure(figsize=(5, 3))
    if color_col is not None and color_col in df.columns:
        c_vals = df[color_col].iloc[: len(scores)]
        scatter = plt.scatter(scores[:, 0], scores[:, 1], c=c_vals, cmap="coolwarm", alpha=0.7)
        plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7)

    plt.title("PCA – Mönster i hälsodata")
    plt.xlabel(f"PC1 ({explained_var[0] * 100:.1f}% varians)")
    plt.ylabel(f"PC2 ({explained_var[1] * 100:.1f}% varians)")
    plt.tight_layout()
    plt.show()


class HealthAnalyzer:
    """Samlar statistiska analyser och visualiseringar för hälsostudiens data.

    Attributes:
        df: DataFrame med hälsodata som ska analyseras.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame]) -> None:
        """Initierar analysklassen med en DataFrame eller filsökväg.

        Args:
            data: Indata som en pandas DataFrame, ``Path`` eller filväg.
        """
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            self.df = pd.read_csv(data)

    def basic_summary(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Returnerar medelvärde, standardavvikelse, min och max.

        Args:
            cols: Lista med kolumner att sammanfatta.

        Returns:
            DataFrame med sammanfattande statistik avrundad till två decimaler.
        """
        if cols is None:
            cols = ["age", "weight", "height", "systolic_bp", "cholesterol"]
        return pd.DataFrame(
            {
                "Medel": self.df[cols].mean(),
                "Std": self.df[cols].std(ddof=1),
                "Min": self.df[cols].min(),
                "Max": self.df[cols].max(),
            }
        ).round(2)

    def bp_ci_95(self) -> Tuple[float, float, float]:
        """Beräknar medelvärde och 95 % konfidensintervall för systoliskt blodtryck.

        Returns:
            En tuple med ``(medel, undre_grans, ovre_grans)``.
        """
        bp = self.df["systolic_bp"].dropna()
        mean_val = float(bp.mean())
        margin = float(stats.norm.ppf(0.975) * (bp.std(ddof=1) / np.sqrt(len(bp))))
        return mean_val, mean_val - margin, mean_val + margin

    def plot_cholesterol_by_disease(self) -> None:
        """Ritar ett stapeldiagram över medelkolesterol per sjukdomsstatus."""
        grouped = self.df.groupby("disease")["cholesterol"].mean()
        plt.figure(figsize=(5, 3))
        grouped.plot(kind="bar", color=["steelblue", "coral"], edgecolor="black")
        plt.title("Medelkolesterol (0 = frisk, 1 = sjuk)")
        plt.xlabel("Sjukdom (disease)")
        plt.ylabel("Medelkolesterol")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def disease_prevalence_by_sex(self) -> pd.DataFrame:
        """Beräknar antal, andel med sjukdom och chi-två-test per kön.

        Returns:
            DataFrame med antal individer, antal sjuka, andel i decimal och procent samt p-värde.
        """
        return disease_prevalence_by_sex(self.df)

    def plot_extended_analysis(self) -> None:
        """Ritar sjukdomsförekomst per kön och blodtryck mot ålder med trendlinje."""
        plot_extended_analysis(self.df)