"""Paper-order regression tests for knowledge-template clustering."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from reta.knowledge.pool import GroundedEntity, GroundedTemplate


HAS_CLUSTERING_DEPS = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("sklearn") is not None
)
if HAS_CLUSTERING_DEPS:
    import numpy as np

    from reta.knowledge.clustering import cluster_templates, l2_normalize


def grounded_template(index: int) -> GroundedTemplate:
    root = f"CCS:ROOT{index}"
    target = f"CCS:TARGET{index}"
    return GroundedTemplate(
        root_concept_id=root,
        root_name=f"Root {index}",
        definition=f"Definition {index}.",
        cascade_entities=[GroundedEntity(target, f"Target {index}", "CCS", 1.0)],
        subgraph_nodes=[root, target],
        subgraph_edges=[(root, target)],
    )


class _FakeClusterer:
    def __init__(self, labels: np.ndarray, seen: list[np.ndarray]):
        self._labels = labels
        self._seen = seen

    def fit_predict(self, values: np.ndarray) -> np.ndarray:
        self._seen.append(np.asarray(values).copy())
        return self._labels.copy()


@unittest.skipUnless(HAS_CLUSTERING_DEPS, "NumPy and scikit-learn are not installed")
class ClusteringOrderTests(unittest.TestCase):
    def test_clusters_and_selects_medoids_before_projecting_centroids(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
        clustering_inputs: list[np.ndarray] = []
        projection_inputs: list[np.ndarray] = []

        class FakeEmbedder:
            def __init__(self, **_kwargs):
                pass

            def encode(self, texts):
                self.last_texts = list(texts)
                return embeddings.copy()

        def fake_clusterer(**_kwargs):
            return _FakeClusterer(labels, clustering_inputs)

        def fake_projection(values: np.ndarray, dim: int) -> np.ndarray:
            projection_inputs.append(np.asarray(values).copy())
            self.assertEqual(dim, 2)
            return np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)

        items = [grounded_template(index) for index in range(4)]
        with (
            patch("reta.knowledge.clustering.TextEmbedder", FakeEmbedder),
            patch("reta.knowledge.clustering.AgglomerativeClustering", fake_clusterer),
            patch("reta.knowledge.clustering.project_to_dim", fake_projection),
        ):
            templates = cluster_templates(items, projection_dim=2)

        normalized = l2_normalize(embeddings, axis=1)
        np.testing.assert_allclose(clustering_inputs[0], normalized, atol=1e-7)

        expected_centroids = np.stack(
            [
                l2_normalize(normalized[:3].mean(axis=0, keepdims=True), axis=1)[0],
                normalized[3],
            ]
        )
        self.assertEqual(len(projection_inputs), 1)
        np.testing.assert_allclose(projection_inputs[0], expected_centroids, atol=1e-7)

        self.assertEqual([template.member_indices for template in templates], [[0, 1, 2], [3]])
        self.assertEqual([template.medoid_idx for template in templates], [1, 3])
        np.testing.assert_allclose(templates[0].vector, [0.6, 0.8], atol=1e-7)
        np.testing.assert_allclose(templates[1].vector, [0.0, 1.0], atol=1e-7)

    def test_single_template_is_projected_after_its_original_space_centroid(self) -> None:
        embedding = np.asarray([[0.0, 3.0, 4.0]], dtype=np.float32)
        projection_inputs: list[np.ndarray] = []

        class FakeEmbedder:
            def __init__(self, **_kwargs):
                pass

            def encode(self, _texts):
                return embedding.copy()

        def fake_projection(values: np.ndarray, dim: int) -> np.ndarray:
            projection_inputs.append(np.asarray(values).copy())
            self.assertEqual(dim, 2)
            return np.asarray([[6.0, 8.0]], dtype=np.float32)

        with (
            patch("reta.knowledge.clustering.TextEmbedder", FakeEmbedder),
            patch("reta.knowledge.clustering.project_to_dim", fake_projection),
        ):
            templates = cluster_templates([grounded_template(0)], projection_dim=2)

        np.testing.assert_allclose(
            projection_inputs[0],
            l2_normalize(embedding, axis=1),
            atol=1e-7,
        )
        self.assertEqual(templates[0].medoid_idx, 0)
        np.testing.assert_allclose(templates[0].vector, [0.6, 0.8], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
