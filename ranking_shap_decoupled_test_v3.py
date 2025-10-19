import numpy as np
import shap
from scipy.stats import kendalltau
import pandas as pd
from functools import partial
from shap.utils._legacy import convert_to_model
from utils.explanation import AttributionExplanation, SelectionExplanation
from utils.helper_functions import rank_list, replace_words_in_sentences, rank_by_original_index
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import random

def placeholder_predict(array):
    Warning(
        "The model.predict function needs to be defined for each query individually."
    )
    return np.array([0] * len(array))



class RankingShapDecoupled:
    """
    Enhanced RankingShap with decoupled pipeline capabilities.
    Minimal modifications to the original class with added functionality for:
    1. Mask generation and saving
    2. Data perturbation using saved masks  
    3. SHAP calculation from perturbed data using internal SHAP mechanism
    """
    
    def __init__(
        self,
        permutation_sampler,
        background_data,
        original_model,
        explanation_size=3,
        nsample_permutations=None,
        rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
        seed=42,
        output_dir="rankingshap_outputs",
        kernel_weight_mode: str = "binom",  # 'gammaln' (log-space) or 'binom'
        l1_reg: Optional[float] = None,
    ):
        assert permutation_sampler in ["kernel", "sampling"]
        self.permutation_sampler = permutation_sampler
        self.background_data = background_data
        self.original_model = original_model
        self.explanation_size = explanation_size
        self.nsamples = nsample_permutations
        self.seed = seed
        self.l1_reg = l1_reg
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])

        self.explainer = self.get_explainer()
        self.rank_similarity_coefficient = rank_similarity_coefficient
        # self.new_model_predict = partial(
        #     new_model_predict_val,
        #     original_model_predict=original_model,
        #     similarity_coefficient=rank_similarity_coefficient,
        #     ranking_shap_instance=self
        # )
        self.feature_attribution_explanation = None
        self.feature_selection_explanation = None
        self.all_masks = []
        
        # New attributes for decoupled functionality
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if kernel_weight_mode not in ("gammaln", "binom"):
            raise ValueError("kernel_weight_mode must be 'gammaln' or 'binom'")
        self.kernel_weight_mode = kernel_weight_mode
        print(f"Using kernel weight mode: {self.kernel_weight_mode}")

    def get_explainer(self):
        if self.permutation_sampler == "kernel":
            shap_explainer = shap.KernelExplainer(
                placeholder_predict, self.background_data, nsamples=self.nsamples
            )
        elif self.permutation_sampler == "sampling":
            shap_explainer = shap.SamplingExplainer(
                placeholder_predict, self.background_data, nsamples=self.nsamples
            )
        return shap_explainer
    
    def generate_masks_only(
        self,
        query_features: List[str],
        query_id: str = "",
        save_masks: bool = True,
        filename_prefix: str = "masks"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Step 1: Generate and optionally save masks without computing SHAP values.
        
        Args:
            query_features: List of feature tokens for the query
            query_id: Identifier for the query
            save_masks: Whether to save masks to file
            filename_prefix: Prefix for saved files
            
        Returns:
            Tuple of (masks array, metadata dictionary)
        """
        print(f"Generating masks for query: {query_id}")
        print(f"Query features: {query_features}")
        
        # Clear previous masks
        self.all_masks = []
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        # Create a dummy model that just collects masks
        # Use constant return for determinism
        def mask_collecting_model(array):
            # print("=="*50)
            # print(array)
            self.all_masks.extend(array.copy())
            # Return constant value - doesn't affect fixed sampling
            return np.ones(len(array))
        
        # Set the mask collecting model
        self.explainer.model = convert_to_model(mask_collecting_model)
        
        # Generate masks by calling SHAP explainer
        vector_of_keeps = np.array([np.full(self.feature_shape, '<keep>')])
        
        if self.permutation_sampler == "kernel":
            _ = self.explainer.shap_values(vector_of_keeps, nsamples=self.nsamples)
        else:
            _ = self.explainer(vector_of_keeps, nsamples=self.nsamples)
        
        # Convert collected masks to numpy array
        masks = np.array(self.all_masks)
        
        # Generate metadata
        metadata = {
            "query_id": query_id,
            "query_features": query_features,
            "feature_shape": self.feature_shape,
            "num_features": self.num_features,
            "num_masks": len(masks),
            "permutation_sampler": self.permutation_sampler,
            "nsample_permutations": self.nsamples,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "background_data_shape": np.array(self.background_data).shape,
            "mask_values": {
                "keep_token": "<keep>",
                "mask_token": "<unk>",
                "description": "Masks where '<keep>' preserves original feature, '<unk>' masks the feature"
            }
        }
        
        if save_masks:
            self.save_masks(masks, metadata, filename_prefix)
        
        print(f"Generated {len(masks)} masks")
        return masks, metadata
    

    
    def apply_masks_to_documents(
        self,
        masks_filepath: str,
        documents: List[str],
        query: str = "",
        save_perturbed: bool = True,
        output_filename: str = None
    ) -> List[Tuple[np.ndarray, List[str], Dict[str, Any]]]:
        """
        Step 2: Apply saved masks to documents and compute perturbation scores.
        
        Args:
            masks_filepath: Path to saved masks .npz file
            documents: List of documents to perturb
            query: Query string for context
            save_perturbed: Whether to save perturbed results
            output_filename: Custom filename for perturbed data
            
        Returns:
            List of (mask, perturbed_documents, perturbation_info) tuples
        """
        # Load masks and metadata
        data = np.load(masks_filepath, allow_pickle=True)
        masks = data['masks']
        query_features = data['query_features'].tolist()

        print(f"Loaded {len(masks)} masks from {masks_filepath}")
        print(f"Query features: {query_features}")
        print(f"Applying to {len(documents)} documents")
        
        # Calculate original ranking
        original_preds = self.original_model(query, documents)
        # original_ranking = rank_list(original_preds)
        original_ranking = rank_by_original_index(original_preds)

        print(f"Original ranking: {original_ranking}")
        
        perturbed_results = []
        

        for mask_idx, mask in enumerate(masks):
            # Identify words to replace
            # Best of both worlds
            words_to_replace = set(
                q for q, m in zip(query_features, mask) if m == "<unk>"
            )
            # print(f"\nMask {mask_idx}: replaced_with_unk = {words_to_replace}")
            # Apply perturbation using existing helper function
            perturbed_docs = replace_words_in_sentences(
                documents, words_to_replace, unk_token="<unk>", case_sensitive=False
            )
            print(perturbed_docs)
            # Calculate perturbation score
            perturbed_preds = self.original_model(query, perturbed_docs)
            # perturbed_ranking = rank_list(perturbed_preds)
            perturbed_ranking = rank_by_original_index(perturbed_preds)
            print("perturbed_ranking:", perturbed_ranking)
            similarity_score = self.rank_similarity_coefficient(original_ranking, perturbed_ranking)
            if hasattr(similarity_score, 'rbo'):
                similarity_score = similarity_score.rbo()
            # print("similarity_score:", similarity_score)
            # Create perturbation info
            perturbation_info = {
                "mask_index": mask_idx,
                "original_mask": mask.tolist(),
                "words_replaced": list(words_to_replace),
                "num_documents": len(documents),
                "perturbation_applied": len(words_to_replace) > 0,
                "similarity_score": float(similarity_score),
                "original_ranking": original_ranking.tolist(),
                "perturbed_ranking": perturbed_ranking.tolist(),
                "perturbed_documents": perturbed_docs # Sample of perturbed documents
            }
            
            perturbed_results.append((mask, perturbed_docs, perturbation_info))
            
            if mask_idx < 10:  # Debug output
                print(f"\nMask {mask_idx}: words_to_replace = {words_to_replace}")
                print(f"  Similarity score: {similarity_score}")
                if len(documents) > 0:
                    for i, orig in enumerate(documents):
                        pert = perturbed_docs[i] if i < len(perturbed_docs) else "<missing>"
                        print(f"  Doc {i}: Original: {orig}")
                        print(f"         Perturbed: {pert}")
        
        if save_perturbed:
            if output_filename is None:
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"perturbed_data.npz"
            
            output_path = os.path.join(self.output_dir, output_filename)
            self._save_perturbed_data(perturbed_results, output_path, {
                "masks_filepath": masks_filepath,
                "query": query,
                "num_documents": len(documents),
                "original_ranking": original_ranking.tolist()
                # "original_metadata": {k: v.item() if hasattr(v, 'item') else v for k, v in data.items() if k != 'masks'}
            })
        
        return perturbed_results
    
    

    def calculate_shap_from_perturbed(
        self,
        perturbed_data_filepath: str,
        query_features: List[str],
        query_id: str = "",
        documents: Optional[List[str]] = None,
        query: Optional[str] = None
    ) -> AttributionExplanation:
        """
        Step 3: Calculate SHAP values using SHAP's kernel weights
        and weighted least squares on precomputed masks and scores.
        """
        print(f"Calculating SHAP values from: {perturbed_data_filepath}")
        
        # Load precomputed data
        data = np.load(perturbed_data_filepath, allow_pickle=True)
        perturbed_data_list = data['perturbed_data']
        
        masks = []
        scores = []
        for item in perturbed_data_list:
            masks.append(np.array(item['mask']))
            scores.append(item['perturbation_info']['similarity_score'])
        
        masks = np.array(masks)
        scores = np.array(scores)
        
        print(f"Loaded {len(masks)} precomputed mask-score pairs")
        
        baseline_score = None
        if documents is not None and query is not None:
            try:
                baseline_score = self._compute_empty_coalition_similarity(
                    documents=documents,
                    query=query,
                    query_features=query_features
                )
                print(f"Empty-coalition baseline (fnull) = {baseline_score:.6f}")
            except Exception as e:
                print(f"WARNING: failed to compute empty-coalition baseline, falling back. Error: {e}")
        
        # Compute SHAP values using SHAP's kernel and weighted regression
        shap_values = self._compute_shap_from_masks_and_scores(
            masks, scores, len(query_features), fnull=baseline_score, fx=1.0
        )

        # Build explanation
        exp_word_dict = {
            query_features[i]: shap_values[i] 
            for i in range(min(len(shap_values), len(query_features)))
        }
        exp_word_list = sorted(exp_word_dict.items(), key=lambda x: x[1], reverse=True)
        
        explanation = AttributionExplanation(
            explanation=exp_word_list,
            num_features=len(query_features),
            query_id=query_id
        )
        
        return explanation


    def _compute_empty_coalition_similarity(
        self,
        documents: List[str],
        query: str,
        query_features: List[str]
    ) -> float:
        """
        Compute similarity score for the empty coalition (all features masked).
        This serves as the SHAP baseline fnull in the ranking-similarity setting.
        """
        # Original ranking
        original_preds = self.original_model(query, documents)
        original_ranking = rank_by_original_index(original_preds)

        # Mask all query features
        words_to_replace = set(query_features)
        all_masked_docs = replace_words_in_sentences(
            documents, words_to_replace, unk_token="<unk>", case_sensitive=False
        )
        # Perturbed ranking under empty coalition
        perturbed_preds = self.original_model(query, all_masked_docs)
        perturbed_ranking = rank_by_original_index(perturbed_preds)

        sim = self.rank_similarity_coefficient(original_ranking, perturbed_ranking)
        if hasattr(sim, 'rbo'):
            sim = sim.rbo()
        return float(sim)



    def _compute_shap_from_masks_and_scores(
        self, 
        masks: np.ndarray, 
        scores: np.ndarray, 
        num_features: int,
        fnull: float | None = None,
        fx: float | None = None,
    ) -> np.ndarray:
        """
        KernelSHAP-compatible solve from precomputed masks and scores.
        Mirrors the SHAP KernelExplainer single-output solve:
        - compute kernel weights for interior coalitions
        - optional L1 feature selection (LARS)
        - eliminate one variable using efficiency constraint
        - weighted least squares with solve -> lstsq fallback
        """
        import numpy as np

        # 0) Deduplicate identical masks; average scores; keep replicate counts as weights
        masks, scores, repeat_counts = self._deduplicate_masks_and_scores(masks, scores)

        # 1) Binary mask matrix and coalition sizes
        maskMatrix = self._convert_masks_to_binary(masks, num_features).astype(float)  # (n, M)
        M = int(num_features)
        s = maskMatrix.sum(axis=1).astype(int)

        # 2) Filter non-finite scores early
        finite = np.isfinite(scores)
        if not np.all(finite):
            maskMatrix = maskMatrix[finite]
            s = s[finite]
            repeat_counts = repeat_counts[finite]
            scores = scores[finite]

        # 3) Infer fnull (s=0) and fx (s=M) from boundary coalitions when present
        idx_empty = np.where(s == 0)[0]
        idx_full = np.where(s == M)[0]
        if fnull is None and idx_empty.size > 0:
            fnull = np.average(scores[idx_empty], weights=repeat_counts[idx_empty])
        if fx is None and idx_full.size > 0:
            fx = np.average(scores[idx_full], weights=repeat_counts[idx_full])
        if fx is None:
            fx = 1.0  # identical ranking similarity
        if fnull is None:
            # Last resort: interior-weighted average as a baseline
            interior_tmp = (s > 0) & (s < M)
            if not np.any(interior_tmp):
                raise ValueError("No interior coalitions available to estimate fnull.")
            w_tmp = self._compute_shapley_kernel_weights(s[interior_tmp], M) * repeat_counts[interior_tmp]
            fnull = np.average(scores[interior_tmp], weights=w_tmp)

        # 4) Adjusted targets (identity link)
        eyAdj = scores - fnull         # eyAdj = link(ey) - link(fnull)
        delta = fx - fnull             # link(fx) - link(fnull)

        # 5) Keep only interior coalitions for the regression
        interior = (s > 0) & (s < M)
        if not np.any(interior):
            raise ValueError("Insufficient interior coalitions (need interior 0 < |S| < M).")
        maskMatrix = maskMatrix[interior]
        eyAdj = eyAdj[interior]
        s = s[interior]
        repeat_counts = repeat_counts[interior]

        # 6) SHAP kernel weights times repeat counts (equivalent to row replication)
        kernelWeights = self._compute_shapley_kernel_weights(s, M)
        kernelWeights = kernelWeights * repeat_counts

        # 7) Optional L1 feature selection (SHAP-style) using augmented, row-weighted system
        nonzero_inds = np.arange(M)
        if hasattr(self, 'l1_reg') and isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
            try:
                from sklearn.linear_model import lars_path
                r = int(self.l1_reg[len("num_features(") : -1])

                # SHAP augments the system to include the constraint
                w_aug = np.hstack((kernelWeights * (M - s), kernelWeights * s))
                w_sqrt_aug = np.sqrt(w_aug)

                eyAdj_aug = np.hstack((eyAdj, eyAdj - delta))
                eyAdj_aug *= w_sqrt_aug

                mask_aug = np.vstack((maskMatrix, maskMatrix - 1.0))
                mask_aug = (w_sqrt_aug[:, None] * mask_aug)

                # LARS path; collect active indices up to r
                _, active, _ = lars_path(mask_aug, eyAdj_aug, max_iter=r)
                nonzero_inds = np.array(active, dtype=int)
                if nonzero_inds.size == 0:
                    nonzero_inds = np.arange(M)
                elif nonzero_inds.size < 2:
                    nonzero_inds = np.concatenate([nonzero_inds, [np.setdiff1d(np.arange(M), nonzero_inds)[0]]])
            except Exception as e:
                print(f"NOTE: L1 skipped: {e}")
                nonzero_inds = np.arange(M)

        # 8) Eliminate one variable with efficiency constraint (exactly as in SHAP)
        if nonzero_inds.size > 1:
            col_var = maskMatrix[:, nonzero_inds].var(axis=0)
            pivot_idx = np.argmax(col_var)
            pivot = nonzero_inds[pivot_idx]
            others = np.delete(nonzero_inds, pivot_idx)
        else:
            pivot = nonzero_inds[0]
            others = np.array([], dtype=int)
        
        eyAdj2 = eyAdj - maskMatrix[:, pivot] * delta
        etmp = (maskMatrix[:, others].T - maskMatrix[:, pivot]).T
        # 9) Weighted least squares solve with SHAP’s fallback strategy
        y = np.asarray(eyAdj2)
        X = etmp
        W = kernelWeights

        # Fast path: normal equations; fallback to lstsq on row-weighted system
        try:
            # WX = W[:, None] * X
            # w_sol = np.linalg.solve(X.T @ WX, WX.T @ y)
            WX = W[:, None] * X
            XtWX = X.T @ WX
            XtWy = WX.T @ y
            alpha = 1e-6 * (np.linalg.norm(XtWX, ord=2) + 1e-12)
            w_sol = np.linalg.solve(XtWX + alpha * np.eye(XtWX.shape[0]), XtWy)
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn(
                "Linear regression equation is singular, a least squares solution is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            sqrt_W = np.sqrt(W)
            w_sol = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]

        # 10) Reconstruct full phi and clean tiny numerical noise
        phi = np.zeros(M)
        phi[others] = w_sol
        phi[pivot] = delta - w_sol.sum()
        phi[np.isclose(phi, 0.0, atol=1e-10)] = 0.0

        # Sanity check for efficiency
        if not np.isclose(phi.sum(), delta, atol=1e-8):
            print(f"WARNING: sum(phi)={phi.sum():.8f} differs from delta={delta:.8f}")

        return phi



    def _compute_shapley_kernel_weights(
        self,
        coalition_sizes,
        num_features: int,
        method: str = None
    ):
        """
        SHAP kernel weights for interior subset sizes:
            w(s) = (M-1) / [C(M,s) * s * (M-s)], for 0 < s < M

        method:
          - 'gammaln' (default): compute in log-space using gammaln for stability
          - 'binom': use scipy.special.binom (fast; fine for moderate M)
        """
        import numpy as np

        M = int(num_features)
        s = np.asarray(coalition_sizes, dtype=int)
        w = np.zeros_like(s, dtype=float)

        interior = (s > 0) & (s < M)
        if not np.any(interior):
            return w

        method = (method or getattr(self, "kernel_weight_mode", "gammaln")).lower()
        print("Using kernel weight method:", method)
        s_i = s[interior].astype(float)

        if method == "binom":
            try:
                from scipy.special import binom
                C = binom(M, s_i)
            except Exception:
                from math import comb as _comb
                C = np.array([float(_comb(M, int(k))) for k in s_i], dtype=float)
            denom = C * s_i * (M - s_i)
            w[interior] = (M - 1.0) / denom
        elif method == "gammaln":
            try:
                from scipy.special import gammaln
                # log C(M, s) = lgamma(M+1) - lgamma(s+1) - lgamma(M-s+1)
                logC = gammaln(M + 1.0) - gammaln(s_i + 1.0) - gammaln(M - s_i + 1.0)
                # log denom = logC + log s + log(M - s)
                log_denom = logC + np.log(s_i) + np.log(M - s_i)
                # log w = log(M-1) - log_denom
                log_w = np.log(M - 1.0) - log_denom
                w[interior] = np.exp(log_w)
            except Exception:
                # Fallback to binom path if SciPy missing
                from math import comb as _comb
                C = np.array([float(_comb(M, int(k))) for k in s_i], dtype=float)
                denom = C * s_i * (M - s_i)
                w[interior] = (M - 1.0) / denom
        else:
            raise ValueError("method must be 'gammaln' or 'binom'")

        return w

    def _deduplicate_masks_and_scores(self, masks, scores):
        """
        Collapse duplicate masks by averaging scores and increasing weights.
        """
        from collections import defaultdict
        
        # Group by mask
        mask_to_indices = defaultdict(list)
        for i, mask in enumerate(masks):
            mask_key = tuple(mask.flatten())
            mask_to_indices[mask_key].append(i)
        
        # Deduplicate
        unique_masks = []
        unique_scores = []
        repeat_counts = []
        
        for mask_key, indices in mask_to_indices.items():
            unique_masks.append(masks[indices[0]])
            # Average scores for duplicates
            unique_scores.append(np.mean([scores[i] for i in indices]))
            repeat_counts.append(len(indices))
        
        if len(unique_masks) < len(masks):
            print(f"Collapsed {len(masks)} masks to {len(unique_masks)} unique masks")
        
        return np.array(unique_masks), np.array(unique_scores), np.array(repeat_counts)      


    def _convert_masks_to_binary(self, masks, num_features):
        """Convert string masks to binary matrix for SHAP computation."""
        mask_matrix = np.zeros((len(masks), num_features))
        
        for i, mask in enumerate(masks):
            for j in range(min(len(mask), num_features)):
                # 1 if feature is kept, 0 if masked
                mask_matrix[i, j] = 1.0 if mask[j] == '<keep>' else 0.0
        
        return mask_matrix

    # UTILITY METHODS
    
    def _save_perturbed_data(self, perturbed_results, filepath, metadata):
        """Save perturbed data to file."""
        save_data = {
            "perturbed_data": [
                {
                    "mask": mask.tolist(),
                    "perturbed_documents": perturbed_docs,
                    "perturbation_info": perturbation_info
                }
                for mask, perturbed_docs, perturbation_info in perturbed_results
            ],
            "metadata": metadata,
            "num_perturbations": len(perturbed_results)
        }
        
        np.savez_compressed(filepath, **save_data)
        print(f"Perturbed data saved to: {filepath}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    # CONVENIENCE METHOD FOR FULL DECOUPLED PIPELINE
    
    def run_decoupled_pipeline(
        self,
        query_features: List[str],
        documents: List[str],
        query: str,
        query_id: str = "",
        save_intermediate: bool = True
    ) -> AttributionExplanation:
        """
        Run the complete decoupled pipeline in sequence.
        
        Args:
            query_features: List of feature tokens for the query
            documents: List of documents to analyze
            query: Query string
            query_id: Query identifier
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Final AttributionExplanation
        """
        print("=== Step 1: Generating Masks ===")
        masks, metadata = self.generate_masks_only(
            query_features, query_id, save_masks=save_intermediate
        )
        
        if save_intermediate:
            # Use the saved file
            timestamp = metadata['timestamp'].replace(':', '').replace('-', '').replace('T', '_')[:15]
            masks_file = os.path.join(self.output_dir, f"masks_{query_id}_{timestamp}.npz")
        else:
            # Save temporarily
            masks_file = os.path.join(self.output_dir, "temp_masks.npz")
            np.savez_compressed(masks_file, masks=masks, **metadata)
        
        print("\n=== Step 2: Applying Masks to Documents and Computing Scores ===")
        perturbed_results = self.apply_masks_to_documents(
            masks_file, documents, query, save_perturbed=save_intermediate
        )
        
        if save_intermediate:
            perturbed_file = os.path.join(self.output_dir, f"perturbed_data_{timestamp}.npz")
        else:
            perturbed_file = os.path.join(self.output_dir, "temp_perturbed.npz")
        
        print("\n=== Step 3: Calculating SHAP Values Using Internal SHAP Mechanism ===")
        explanation = self.calculate_shap_from_perturbed(
            perturbed_file, query_features, query_id
        )
        
        # Clean up temporary files if not saving intermediate results
        if not save_intermediate:
            for temp_file in [masks_file, perturbed_file]:
                if os.path.exists(temp_file) and "temp_" in temp_file:
                    os.remove(temp_file)
        
        return explanation
    


    def save_masks(
        self,
        masks: np.ndarray,
        metadata: Dict[str, Any],
        filename_prefix: str = "masks"
    ) -> str:
        """Save masks and metadata to files."""
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_id = metadata.get("query_id", "unknown")
        # base_filename = f"{filename_prefix}_{query_id}_{timestamp}"
        base_filename = f"{filename_prefix}_{query_id}"
        
        # Save masks as .npz file
        masks_file = os.path.join(self.output_dir, f"{base_filename}.npz")
        np.savez_compressed(masks_file, masks=masks, **metadata)
        
        # Save metadata as JSON
        metadata_file = os.path.join(self.output_dir, f"{base_filename}_metadata.json")
        json_metadata = self._convert_numpy_types(metadata)
        with open(metadata_file, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        print(f"Masks saved to: {masks_file}")
        print(f"Metadata saved to: {metadata_file}")
        
        return base_filename
             