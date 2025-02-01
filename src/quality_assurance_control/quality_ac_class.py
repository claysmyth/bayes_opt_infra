from src.utils import load_funcs
import polars as pl
import warnings
from typing import Dict, Optional

class QualityAC:
    """
    A class to perform Quality Assurance and Quality Control checks on session data.
    
    Quality Assurance (QA) checks verify if session settings are as expected.
    Quality Control (QC) checks verify if session data meets quality standards.
    
    Functions for both QA and QC are defined in their respective modules and 
    configured via the quality_ac_config.
    """

    def __init__(self, qac_config: Dict):
        self.config = qac_config
        
        # Initialize QA checks
        self.qa_funcs = {}
        if self.config.get("qa_functions"):
            self.qa_funcs = load_funcs(self.config["qa_functions"], "quality_assurance")
            
        # Initialize QC checks
        self.qc_funcs = {}
        if self.config.get("qc_functions"):
            self.qc_funcs = load_funcs(self.config["qc_functions"], "quality_control")

    def quality_assurance_check(self, session_data: pl.DataFrame) -> Optional[bool]:
        """
        Run all configured quality assurance checks on the session data.
        Returns:
            True if any QA check fails (bad data)
            False if all checks pass
            None if no QA checks are configured
        """
        if not self.qa_funcs:
            return None
            
        for func_name, qa_func in self.qa_funcs.items():
            try:
                check_failed = qa_func(session_data)
                if check_failed:
                    warnings.warn(f"Quality Assurance check '{func_name}' failed")
                    return True
            except Exception as e:
                warnings.warn(f"Error in QA check '{func_name}': {str(e)}")
                return True
                
        return False

    def quality_control_check(self, session_data: pl.DataFrame) -> Optional[bool]:
        """
        Run all configured quality control checks on the session data.
        Returns:
            True if any QC check fails (insufficient data)
            False if all checks pass
            None if no QC checks are configured
        """
        if not self.qc_funcs:
            return None
            
        for func_name, qc_func in self.qc_funcs.items():
            try:
                check_failed = qc_func(session_data)
                if check_failed:
                    warnings.warn(f"Quality Control check '{func_name}' failed")
                    return True
            except Exception as e:
                warnings.warn(f"Error in QC check '{func_name}': {str(e)}")
                return True
                
        return False