"""
data_validation.py
------------------
Data validation layer for the End-to-End Churn Prediction pipeline.

Responsibilities:
1. Verify dataset file existence.
2. Load raw/ingested data.
3. Validate schema integrity (detect missing and extra columns).
4. Validate feature data types (numerical and categorical consistency).
5. Analyze missing value ratios against configurable threshold boundaries.
6. Validate target column presence and class label validity.
7. Automatically export a structured validation report (JSON).

Design Principles:
- Pure Python exceptions (no logging dependencies).
- Modular and reusable structure for seamless ZenML pipeline integration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class DataValidator:
    """
    Main Data Validation class for MLOps production pipelines.
    
    Attributes:
        EXPECTED_COLUMNS (List[str]): Default baseline list of expected column names.
        NUMERICAL_FEATURES (List[str]): Default list of expected numerical features.
        CATEGORICAL_FEATURES (List[str]): Default list of expected categorical features.
        TARGET_COLUMN (str): Default target variable name.
        ALLOWED_TARGET_VALUES (List[Any]): Valid target values.
        MISSING_THRESHOLD (float): Maximum tolerable fraction of missing values per column.
    """

    EXPECTED_COLUMNS: List[str] = [
        "customer_id",
        "gender",
        "tenure",
        "monthly_charges",
        "churn",
    ]

    NUMERICAL_FEATURES: List[str] = [
        "age",
        "tenure",
        "monthly_charges",
    ]

    CATEGORICAL_FEATURES: List[str] = [
        "gender",
        "contract",
    ]

    TARGET_COLUMN: str = "churn"
    ALLOWED_TARGET_VALUES: List[Any] = [0, 1]
    MISSING_THRESHOLD: float = 0.50  # 50% threshold

    def __init__(
        self,
        data_path: Union[str, Path] = "data/raw/Churn_Modelling.csv",
        report_path: Union[str, Path] = "artifacts/validation/validation_report.json",
        expected_columns: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        allowed_target_values: Optional[List[Any]] = None,
        missing_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialise le validateur.

        Parameters:
            data_path: chemin vers le dataset
            report_path: chemin du rapport de validation
            expected_columns: liste optionnelle des colonnes attendues
            numerical_features: liste optionnelle des variables numériques
            categorical_features: liste optionnelle des variables catégorielles
            target_column: nom de la colonne cible
            allowed_target_values: valeurs acceptées pour la cible
            missing_threshold: seuil maximal de valeurs manquantes par colonne (0.0 à 1.0)
        """
        self.data_path: Path = Path(data_path)
        self.report_path: Path = Path(report_path)

        self.expected_columns: List[str] = expected_columns or self.EXPECTED_COLUMNS
        self.numerical_features: List[str] = numerical_features or self.NUMERICAL_FEATURES
        self.categorical_features: List[str] = categorical_features or self.CATEGORICAL_FEATURES
        self.target_column: str = target_column or self.TARGET_COLUMN
        self.allowed_target_values: List[Any] = allowed_target_values or self.ALLOWED_TARGET_VALUES
        self.missing_threshold: float = (
            missing_threshold if missing_threshold is not None else self.MISSING_THRESHOLD
        )

    def validate_file_exists(self) -> Path:
        """
        Vérifier que le fichier dataset existe.

        Returns:
            Path: Le chemin du fichier s'il existe.

        Raises:
            FileNotFoundError: Si le fichier est absent.
        """
        if not self.data_path.exists() or not self.data_path.is_file():
            raise FileNotFoundError(f"Le fichier dataset est introuvable : '{self.data_path}'")
        return self.data_path

    def load_data(self) -> pd.DataFrame:
        """
        Charger le dataset avec pandas.

        Returns:
            pd.DataFrame: Le DataFrame chargé.

        Raises:
            FileNotFoundError: Si le fichier est absent.
            ValueError: Si le fichier est vide.
        """
        self.validate_file_exists()
        df: pd.DataFrame = pd.read_csv(self.data_path)

        if df.empty:
            raise ValueError(f"Le jeu de données chargé depuis '{self.data_path}' est vide.")

        return df

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Vérifier que le dataset respecte le schéma attendu.

        Parameters:
            df (pd.DataFrame): DataFrame à valider.

        Returns:
            Dict[str, List[str]]: Dictionnaire contenant les colonnes manquantes et supplémentaires.

        Raises:
            ValueError: Si des colonnes obligatoires sont manquantes.
        """
        missing_columns: List[str] = [col for col in self.expected_columns if col not in df.columns]
        extra_columns: List[str] = [col for col in df.columns if col not in self.expected_columns]

        if missing_columns:
            raise ValueError(
                f"Validation du schéma échouée. Colonnes manquantes : {missing_columns}"
            )

        return {
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
        }

    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Vérifier les types des variables (numériques et catégorielles).

        Parameters:
            df (pd.DataFrame): DataFrame à valider.

        Returns:
            Dict[str, str]: Mappage des colonnes et de leurs dtypes détectés.

        Raises:
            TypeError: Si une variable ne correspond pas au type attendu.
        """
        detected_types: Dict[str, str] = {}

        # Validation des variables numériques
        for col in self.numerical_features:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise TypeError(
                        f"Incohérence de type pour la colonne numérique '{col}' : "
                        f"attendu numérique, reçu '{df[col].dtype}'"
                    )
                detected_types[col] = str(df[col].dtype)

        # Validation des variables catégorielles
        for col in self.categorical_features:
            if col in df.columns:
                is_cat = (
                    pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_string_dtype(df[col])
                    or pd.api.types.is_categorical_dtype(df[col])
                )
                if not is_cat:
                    raise TypeError(
                        f"Incohérence de type pour la colonne catégorielle '{col}' : "
                        f"attendu texte/catégorie, reçu '{df[col].dtype}'"
                    )
                detected_types[col] = str(df[col].dtype)

        return detected_types

    def validate_missing_values(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyser le nombre et le pourcentage de valeurs manquantes par colonne.

        Parameters:
            df (pd.DataFrame): DataFrame à valider.

        Returns:
            Dict[str, Dict[str, Any]]: Métriques des valeurs manquantes par colonne.

        Raises:
            ValueError: Si une colonne dépasse le seuil configurable de valeurs manquantes.
        """
        total_rows: int = len(df)
        missing_summary: Dict[str, Dict[str, Any]] = {}
        exceeded_columns: List[str] = []

        for col in df.columns:
            missing_count: int = int(df[col].isnull().sum())
            missing_pct: float = float(missing_count / total_rows) if total_rows > 0 else 0.0

            missing_summary[col] = {
                "count": missing_count,
                "percentage": round(missing_pct * 100, 2),
            }

            if missing_pct > self.missing_threshold:
                exceeded_columns.append(
                    f"'{col}' ({missing_pct * 100:.2f}% > {self.missing_threshold * 100:.2f}%)"
                )

        if exceeded_columns:
            raise ValueError(
                f"Validation des valeurs manquantes échouée. Colonnes dépassant le seuil : "
                + ", ".join(exceeded_columns)
            )

        return missing_summary

    def validate_target_column(self, df: pd.DataFrame) -> bool:
        """
        Valider la variable cible (présence et valeurs autorisées).

        Parameters:
            df (pd.DataFrame): DataFrame à valider.

        Returns:
            bool: True si la cible est valide.

        Raises:
            ValueError: Si la colonne cible est absente ou contient des classes non autorisées.
        """
        if self.target_column not in df.columns:
            raise ValueError(
                f"La colonne cible '{self.target_column}' est absente du jeu de données."
            )

        unique_values: set = set(df[self.target_column].dropna().unique())
        allowed_set: set = set(self.allowed_target_values)

        invalid_values: set = unique_values - allowed_set
        if invalid_values:
            raise ValueError(
                f"La colonne cible '{self.target_column}' contient des valeurs non autorisées : "
                f"{list(invalid_values)}. Valeurs autorisées : {self.allowed_target_values}"
            )

        return True

    def generate_validation_report(self, results: Dict[str, Any]) -> Path:
        """
        Générer automatiquement un fichier de rapport JSON.

        Parameters:
            results (Dict[str, Any]): Dictionnaire contenant le bilan de validation.

        Returns:
            Path: Le chemin vers le fichier de rapport généré.
        """
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        report_content = {
            "status": results.get("status", "failed"),
            "missing_columns": results.get("missing_columns", []),
            "extra_columns": results.get("extra_columns", []),
            "missing_values": results.get("missing_values", {}),
            "errors": results.get("errors", []),
        }

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report_content, f, indent=4, ensure_ascii=False)

        return self.report_path

    def validate(self) -> bool:
        """
        Méthode principale exécutant la séquence complète de validation :
        validate_file_exists() -> load_data() -> validate_schema() ->
        validate_data_types() -> validate_missing_values() ->
        validate_target_column() -> generate_validation_report()

        Returns:
            bool: True si toutes les validations réussissent, False sinon.
        """
        results: Dict[str, Any] = {
            "status": "passed",
            "missing_columns": [],
            "extra_columns": [],
            "missing_values": {},
            "errors": [],
        }

        try:
            self.validate_file_exists()
            df: pd.DataFrame = self.load_data()

            schema_info = self.validate_schema(df)
            results["missing_columns"] = schema_info["missing_columns"]
            results["extra_columns"] = schema_info["extra_columns"]

            self.validate_data_types(df)
            results["missing_values"] = self.validate_missing_values(df)
            self.validate_target_column(df)

        except Exception as exc:
            results["status"] = "failed"
            results["errors"].append(str(exc))

        self.generate_validation_report(results)
        return results["status"] == "passed"


if __name__ == "__main__":
    import tempfile

    print("--- Test local de DataValidator ---")

    # 1. Création d'un dataset de test temporaire conforme au schéma
    sample_data = pd.DataFrame(
        {
            "customer_id": ["C101", "C102", "C103"],
            "gender": ["Female", "Male", "Female"],
            "tenure": [12, 24, 6],
            "monthly_charges": [65.5, 80.0, 45.2],
            "churn": [0, 1, 0],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        dataset_file = tmp_path / "sample_churn.csv"
        report_file = tmp_path / "artifacts" / "validation" / "validation_report.json"

        sample_data.to_csv(dataset_file, index=False)

        # Initialisation du validateur avec le dataset d'exemple
        validator = DataValidator(
            data_path=dataset_file,
            report_path=report_file,
        )

        is_valid = validator.validate()

        print(f"Statut de la validation : {'RÉUSSIE' if is_valid else 'ÉCHOUÉE'}")
        print(f"Rapport généré dans : {report_file}")

        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as rf:
                print("\nContenu du rapport JSON :")
                print(rf.read())
