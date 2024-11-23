import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, log_loss
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

@dataclass
class AdImpression:
    """Represents a single ad impression"""
    ad_id: str
    position: int
    predicted_ctr: float
    predicted_cvr: float
    bid_amount: float
    actual_click: bool = False
    actual_conversion: bool = False
    user_id: str = ""
    timestamp: datetime = None
    cost: float = 0.0
    
@dataclass
class ExperimentConfig:
    """Configuration for evaluation experiment"""
    name: str
    start_date: datetime
    end_date: datetime
    control_group: str
    treatment_group: str
    metrics: List[str]

class AdsRankingEvaluator:
    """Framework for evaluating ads ranking systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_offline_metrics(self, impressions: List[AdImpression]) -> Dict[str, float]:
        """Calculate offline evaluation metrics"""
        metrics = {}
        
        # Prepare arrays for metric calculations
        y_true_clicks = np.array([imp.actual_click for imp in impressions])
        y_pred_clicks = np.array([imp.predicted_ctr for imp in impressions])
        y_true_conv = np.array([imp.actual_conversion for imp in impressions])
        y_pred_conv = np.array([imp.predicted_cvr for imp in impressions])
        
        # Click prediction metrics
        metrics['ctr_auc'] = roc_auc_score(y_true_clicks, y_pred_clicks)
        metrics['ctr_log_loss'] = log_loss(y_true_clicks, y_pred_clicks)
        
        # Conversion prediction metrics
        clicked_mask = y_true_clicks == 1
        if np.any(clicked_mask):
            metrics['cvr_auc'] = roc_auc_score(
                y_true_conv[clicked_mask], 
                y_pred_conv[clicked_mask]
            )
            metrics['cvr_log_loss'] = log_loss(
                y_true_conv[clicked_mask], 
                y_pred_conv[clicked_mask]
            )
        
        # Position-based metrics
        metrics['average_position'] = np.mean([imp.position for imp in impressions])
        
        # Revenue metrics
        total_cost = sum(imp.cost for imp in impressions)
        total_clicks = sum(y_true_clicks)
        total_conversions = sum(y_true_conv)
        
        metrics['cpc'] = total_cost / total_clicks if total_clicks > 0 else 0
        metrics['cpa'] = total_cost / total_conversions if total_conversions > 0 else 0
        metrics['revenue'] = total_cost
        
        return metrics
    
    def calculate_online_metrics(
        self,
        control_impressions: List[AdImpression],
        treatment_impressions: List[AdImpression]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate online A/B test metrics with statistical significance"""
        
        def calculate_group_metrics(impressions: List[AdImpression]) -> Dict[str, float]:
            total_impressions = len(impressions)
            if total_impressions == 0:
                return {}
                
            clicks = sum(1 for imp in impressions if imp.actual_click)
            conversions = sum(1 for imp in impressions if imp.actual_conversion)
            revenue = sum(imp.cost for imp in impressions)
            
            return {
                'impressions': total_impressions,
                'ctr': clicks / total_impressions,
                'cvr': conversions / clicks if clicks > 0 else 0,
                'revenue_per_impression': revenue / total_impressions,
                'average_position': np.mean([imp.position for imp in impressions])
            }
        
        control_metrics = calculate_group_metrics(control_impressions)
        treatment_metrics = calculate_group_metrics(treatment_impressions)
        
        # Calculate relative differences and confidence intervals
        results = {
            'control': control_metrics,
            'treatment': treatment_metrics,
            'relative_diff': {}
        }
        
        for metric in control_metrics:
            if control_metrics[metric] != 0:
                relative_diff = (treatment_metrics[metric] - control_metrics[metric]) / control_metrics[metric]
                results['relative_diff'][metric] = relative_diff
                
        return results
    
    def calculate_user_metrics(self, impressions: List[AdImpression]) -> Dict[str, float]:
        """Calculate user-centric metrics"""
        user_impressions: Dict[str, List[AdImpression]] = {}
        
        # Group impressions by user
        for imp in impressions:
            if imp.user_id not in user_impressions:
                user_impressions[imp.user_id] = []
            user_impressions[imp.user_id].append(imp)
            
        metrics = {}
        
        # Average impressions per user
        metrics['avg_impressions_per_user'] = np.mean([
            len(imps) for imps in user_impressions.values()
        ])
        
        # User engagement rate (users with at least one click)
        engaged_users = sum(1 for imps in user_impressions.values() 
                          if any(imp.actual_click for imp in imps))
        metrics['user_engagement_rate'] = engaged_users / len(user_impressions)
        
        # User conversion rate (users with at least one conversion)
        converted_users = sum(1 for imps in user_impressions.values()
                            if any(imp.actual_conversion for imp in imps))
        metrics['user_conversion_rate'] = converted_users / len(user_impressions)
        
        return metrics
    
    def calculate_diversity_metrics(self, impressions: List[AdImpression]) -> Dict[str, float]:
        """Calculate diversity and coverage metrics"""
        metrics = {}
        
        # Unique ads shown
        unique_ads = len(set(imp.ad_id for imp in impressions))
        metrics['unique_ads_shown'] = unique_ads
        
        # Position diversity (entropy of position distribution)
        position_counts = {}
        for imp in impressions:
            position_counts[imp.position] = position_counts.get(imp.position, 0) + 1
            
        position_probs = [count/len(impressions) for count in position_counts.values()]
        metrics['position_entropy'] = -sum(p * np.log(p) for p in position_probs)
        
        return metrics
    
    def evaluate_experiment(
        self,
        config: ExperimentConfig,
        control_impressions: List[AdImpression],
        treatment_impressions: List[AdImpression]
    ) -> Dict[str, Dict[str, float]]:
        """Run full evaluation for an A/B test experiment"""
        
        # Filter impressions by date range
        def filter_by_date(impressions: List[AdImpression]) -> List[AdImpression]:
            return [
                imp for imp in impressions
                if config.start_date <= imp.timestamp <= config.end_date
            ]
            
        control_filtered = filter_by_date(control_impressions)
        treatment_filtered = filter_by_date(treatment_impressions)
        
        results = {
            'experiment_info': {
                'name': config.name,
                'start_date': config.start_date.isoformat(),
                'end_date': config.end_date.isoformat(),
                'control_size': len(control_filtered),
                'treatment_size': len(treatment_filtered)
            }
        }
        
        # Online metrics
        results['online_metrics'] = self.calculate_online_metrics(
            control_filtered, treatment_filtered
        )
        
        # Offline metrics for both groups
        results['offline_metrics'] = {
            'control': self.calculate_offline_metrics(control_filtered),
            'treatment': self.calculate_offline_metrics(treatment_filtered)
        }
        
        # User metrics
        results['user_metrics'] = {
            'control': self.calculate_user_metrics(control_filtered),
            'treatment': self.calculate_user_metrics(treatment_filtered)
        }
        
        # Diversity metrics
        results['diversity_metrics'] = {
            'control': self.calculate_diversity_metrics(control_filtered),
            'treatment': self.calculate_diversity_metrics(treatment_filtered)
        }
        
        return results

class MetricSignificanceTester:
    """Statistical significance testing for metrics"""
    
    @staticmethod
    def calculate_confidence_interval(
        metric_control: List[float],
        metric_treatment: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the difference between two groups"""
        control_mean = np.mean(metric_control)
        treatment_mean = np.mean(metric_treatment)
        
        control_var = np.var(metric_control, ddof=1)
        treatment_var = np.var(metric_treatment, ddof=1)
        
        pooled_se = np.sqrt(control_var/len(metric_control) + treatment_var/len(metric_treatment))
        
        # Using normal distribution approximation
        z_score = 1.96  # 95% confidence level
        margin_of_error = z_score * pooled_se
        
        return (
            (treatment_mean - control_mean) - margin_of_error,
            (treatment_mean - control_mean) + margin_of_error
        )

# Example usage
def main():
    # Create sample data
    base_time = datetime.now()
    sample_impressions = [
        AdImpression(
            ad_id=f"ad_{i}",
            position=i % 3 + 1,
            predicted_ctr=0.1,
            predicted_cvr=0.02,
            bid_amount=1.0,
            actual_click=i % 5 == 0,
            actual_conversion=i % 20 == 0,
            user_id=f"user_{i % 100}",
            timestamp=base_time + timedelta(hours=i),
            cost=0.5 if i % 5 == 0 else 0.0
        )
        for i in range(1000)
    ]
    
    # Split into control and treatment
    control = sample_impressions[:500]
    treatment = sample_impressions[500:]
    
    # Create experiment config
    config = ExperimentConfig(
        name="ranking_algorithm_test",
        start_date=base_time,
        end_date=base_time + timedelta(days=7),
        control_group="baseline",
        treatment_group="new_algorithm",
        metrics=["ctr", "cvr", "revenue"]
    )
    
    # Run evaluation
    evaluator = AdsRankingEvaluator()
    results = evaluator.evaluate_experiment(config, control, treatment)
    
    # Print results
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
