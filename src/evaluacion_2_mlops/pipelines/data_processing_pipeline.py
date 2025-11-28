from kedro.pipeline import Pipeline, node
from ..nodes.preprocessing import create_primary_table

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=create_primary_table,
            inputs=["customer_profile", "products", "purchase_history"], # Nombres del catalog.yml
            outputs="primary_data",                                     # Nombre del catalog.yml
            name="create_primary_table_node"
        )
    ])