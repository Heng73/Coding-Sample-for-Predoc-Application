/*==============================================================================
* MASTER ANALYSIS STRUCTURE - EVENT STUDY ANALYSIS
* 
* Purpose: 
*   Comprehensive event study analysis of emission data
*   Replicates AEJ Applied strategy with distance-based treatment
*
* Author: Heng Tang
* Date: 07/03/2025
* 
* File Structure:
*   1. 01_emission_analysis_main.do     - Main emission analysis (all datasets)
*   2. 02_abatement_effort_analysis.do  - Abatement effort analysis  
*   3. 03_industry_analysis.do          - Industry-specific analysis
*==============================================================================*/

/*==============================================================================
* FILE 1: 01_emission_analysis_main.do
* 
* Content:
*   - Full dataset analysis (merged_full.dta)
*   - Reduced dataset analysis (merged.dta) 
*   - Extended dataset analysis (merged_extended.dta)
*   - Balanced panel analysis
*   - All emission variables (SO2, NOx, COD, NH3N, industrial output)
*   - Both SA and PPML estimations
*==============================================================================*/

global data    "/Users/tangheng/Dropbox/Green TFP China/Data"
global results "/Users/tangheng/Dropbox/Green TFP China/RA_Heng_Work/Result"
global workdata "/Users/tangheng/Dropbox/Green TFP China/RA_Heng_Work/Data/Workdata"

/*------------------------------------------------------------------------------
* SECTION 1: DEFINE PLOTTING PROGRAMS
*------------------------------------------------------------------------------*/

// Program for PPML event study plots
capture program drop plot_event_study_ppml
program define plot_event_study_ppml
    syntax, graph_name(string) graph_title(string) [subfolder(string)]
    
    if "`subfolder'" == "" local subfolder "main"
    
    event_plot ols ols ols ols, ///
        stub_lag(dis_1_D_lead_# dis_2_D_lead_# dis_3_D_lead_# dis_4_D_lead_#) ///
        stub_lead(dis_1_D_lag_# dis_2_D_lag_# dis_3_D_lag_# dis_4_D_lag_#) ///
        plottype(scatter) ciplottype(rcap) ///
        together perturb(-0.1(0.06)0.1) trimlead(6) noautolegend ///
        graph_opt(title("`graph_title'", size(medlarge)) ///
            xtitle("Years since the event") ytitle("PPML Coefficient") ///
            xlabel(-5(1)6) ///
            legend(order(1 "0-5km" 3 "5-10km" 5 "10-15km" 7 "15-20km") ///
                   position(6) rows(1) region(style(none))) ///
            xline(-1, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) ///
            graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal))) ///
        lag_opt1(msymbol(O) color(navy%80)) lag_ci_opt1(color(navy%80)) ///
        lag_opt2(msymbol(O) color(cranberry%50)) lag_ci_opt2(color(cranberry%50)) ///
        lag_opt3(msymbol(O) color(forest_green%50)) lag_ci_opt3(color(forest_green%50)) ///
        lag_opt4(msymbol(O) color(orange%50)) lag_ci_opt4(color(orange%50))

    graph export "$results/Reduced_form_evidence/replicate_AEJ/08_18_without_Jiangsu/`subfolder'/`graph_name'_ppml.pdf", replace
end

// Program for PPML event study plots with industry-year FE
capture program drop plot_event_study_ppml_IYFE
program define plot_event_study_ppml_IYFE
    syntax, graph_name(string) graph_title(string) [subfolder(string)]
    
    if "`subfolder'" == "" local subfolder "main"
    
    event_plot ols ols ols ols, ///
        stub_lag(dis_1_D_lead_# dis_2_D_lead_# dis_3_D_lead_# dis_4_D_lead_#) ///
        stub_lead(dis_1_D_lag_# dis_2_D_lag_# dis_3_D_lag_# dis_4_D_lag_#) ///
        plottype(scatter) ciplottype(rcap) ///
        together perturb(-0.1(0.06)0.1) trimlead(6) noautolegend ///
        graph_opt(title("`graph_title' (industry-year FE)", size(medlarge)) ///
            xtitle("Years since the event") ytitle("PPML Coefficient") ///
            xlabel(-5(1)6) ///
            legend(order(1 "0-5km" 3 "5-10km" 5 "10-15km" 7 "15-20km") ///
                   position(6) rows(1) region(style(none))) ///
            xline(-1, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) ///
            graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal))) ///
        lag_opt1(msymbol(O) color(navy%80)) lag_ci_opt1(color(navy%80)) ///
        lag_opt2(msymbol(O) color(cranberry%50)) lag_ci_opt2(color(cranberry%50)) ///
        lag_opt3(msymbol(O) color(forest_green%50)) lag_ci_opt3(color(forest_green%50)) ///
        lag_opt4(msymbol(O) color(orange%50)) lag_ci_opt4(color(orange%50))

    graph export "$results/Reduced_form_evidence/replicate_AEJ/08_18_without_Jiangsu/`subfolder'/`graph_name'_ppml_IYFE.pdf", replace
end

// Program for Sun-Abraham event study plots
capture program drop plot_event_study_SA
program define plot_event_study_SA
    syntax, graph_name(string) graph_title(string) [subfolder(string)]
    
    if "`subfolder'" == "" local subfolder "main"
	
    event_plot e(b_iw)#e(V_iw) e(b_iw)#e(V_iw) e(b_iw)#e(V_iw) e(b_iw)#e(V_iw), ///
        stub_lag(dis_1_D_lead_# dis_2_D_lead_# dis_3_D_lead_# dis_4_D_lead_#) ///
        stub_lead(dis_1_D_lag_# dis_2_D_lag_# dis_3_D_lag_# dis_4_D_lag_#) ///
        plottype(scatter) ciplottype(rcap) ///
        together perturb(-0.1(0.06)0.1) trimlead(6) noautolegend ///
        graph_opt(title("`graph_title'", size(medlarge)) ///
            xtitle("Years since the event") ytitle("SA Coefficient") ///
            xlabel(-5(1)6) ///
            legend(order(1 "0-5km" 3 "5-10km" 5 "10-15km" 7 "15-20km") ///
                   position(6) rows(1) region(style(none))) ///
            xline(-1, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) ///
            graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal))) ///
        lag_opt1(msymbol(O) color(navy%80)) lag_ci_opt1(color(navy%80)) ///
        lag_opt2(msymbol(O) color(cranberry%50)) lag_ci_opt2(color(cranberry%50)) ///
        lag_opt3(msymbol(O) color(forest_green%50)) lag_ci_opt3(color(forest_green%50)) ///
        lag_opt4(msymbol(O) color(orange%50)) lag_ci_opt4(color(orange%50))

    graph export "$results/Reduced_form_evidence/replicate_AEJ/08_18_without_Jiangsu/`subfolder'/`graph_name'_SA.pdf", replace
end

// Program for Sun-Abraham event study plots with industry-year FE
capture program drop plot_event_study_SA_IYFE
program define plot_event_study_SA_IYFE
    syntax, graph_name(string) graph_title(string) [subfolder(string)]
    
    if "`subfolder'" == "" local subfolder "main"
	
    event_plot e(b_iw)#e(V_iw) e(b_iw)#e(V_iw) e(b_iw)#e(V_iw) e(b_iw)#e(V_iw), ///
        stub_lag(dis_1_D_lead_# dis_2_D_lead_# dis_3_D_lead_# dis_4_D_lead_#) ///
        stub_lead(dis_1_D_lag_# dis_2_D_lag_# dis_3_D_lag_# dis_4_D_lag_#) ///
        plottype(scatter) ciplottype(rcap) ///
        together perturb(-0.1(0.06)0.1) trimlead(5) noautolegend ///
        graph_opt(title("`graph_title' (industry-year FE)", size(medlarge)) ///
            xtitle("Years since the event") ytitle("SA Coefficient") ///
            xlabel(-5(1)6) ///
            legend(order(1 "0-5km" 3 "5-10km" 5 "10-15km" 7 "15-20km") ///
                   position(6) rows(1) region(style(none))) ///
            xline(-1, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) ///
            graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal))) ///
        lag_opt1(msymbol(O) color(navy%80)) lag_ci_opt1(color(navy%80)) ///
        lag_opt2(msymbol(O) color(cranberry%50)) lag_ci_opt2(color(cranberry%50)) ///
        lag_opt3(msymbol(O) color(forest_green%50)) lag_ci_opt3(color(forest_green%50)) ///
        lag_opt4(msymbol(O) color(orange%50)) lag_ci_opt4(color(orange%50))

    graph export "$results/Reduced_form_evidence/replicate_AEJ/08_18_without_Jiangsu/`subfolder'/`graph_name'_SA_IYFE.pdf", replace
end

/*------------------------------------------------------------------------------
* SECTION 2: DATA PREPARATION PROGRAM (SHARED LOGIC)
*------------------------------------------------------------------------------*/

capture program drop prepare_event_data
program define prepare_event_data
    // Rename and clean industry variable
    rename indus bindustry
    drop if bindustry == .

    // Generate event time variables
    gen event_time = .
    replace event_time = year - openyear if !missing(openyear)

    // Create event time dummy variables
    gen D_lag_5  = (event_time <= -5 & event_time != .)
    gen D_lag_4  = (event_time == -4 & event_time != .)
    gen D_lag_3  = (event_time == -3 & event_time != .)
    gen D_lag_2  = (event_time == -2 & event_time != .)
    gen D_lag_1  = (event_time == -1 & event_time != .)
    gen D_lead_0 = (event_time == 0 & event_time != .)
    gen D_lead_1 = (event_time == 1 & event_time != .)
    gen D_lead_2 = (event_time == 2 & event_time != .)
    gen D_lead_3 = (event_time == 3 & event_time != .)
    gen D_lead_4 = (event_time == 4 & event_time != .)
    gen D_lead_5 = (event_time == 5 & event_time != .)
    gen D_lead_6 = (event_time >= 6 & event_time != .)
    replace D_lag_1 = 0  // Reference period

    // Create distance bands
    gen dis_1 = (distance_km < 5 & distance_km != .)
    gen dis_2 = (distance_km >= 5 & distance_km < 10 & distance_km != .)
    gen dis_3 = (distance_km >= 10 & distance_km < 15 & distance_km != .)
    gen dis_4 = (distance_km >= 15 & distance_km < 20 & distance_km != .)

    // Create lastcohort dummy for SA estimation
    sum openyear
    gen lastcohort = (distance_km > 20)

    // Generate interaction terms
    foreach var in D_lag_5 D_lag_4 D_lag_3 D_lag_2 D_lag_1 D_lead_0 D_lead_1 D_lead_2 D_lead_3 D_lead_4 D_lead_5 D_lead_6 {
        gen dis_1_`var' = dis_1 * `var'
        gen dis_2_`var' = dis_2 * `var'
        gen dis_3_`var' = dis_3 * `var'
        gen dis_4_`var' = dis_4 * `var'
    }

    // Clean and transform emission variables
    foreach var in industrial_output wastewater_emission so2_emission nox_emission cod_emission nh3n_emission {
        replace `var' = . if `var' < 0
        gen log_`var' = log(1 + `var')
    }

    // Generate industry classification
    gen six_industry = (bindustry == 44 | bindustry == 30 | bindustry == 31 | bindustry == 26 | bindustry == 32 | bindustry == 22)
end


/*------------------------------------------------------------------------------
* SECTION 3: 2008-2018 DATASET ANALYSIS (merged.dta)
*------------------------------------------------------------------------------*/

use "$workdata/merged.dta", clear
prepare_event_data


// Main emission analysis with firm-year trends
local emission_vars "log_so2_emission log_nox_emission log_cod_emission log_nh3n_emission"
foreach var of local emission_vars {
    // With firm-year trends
    eventstudyinteract `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        vce(cluster city_code) absorb(org_ID_num#c.year year#monitor_id) cohort(openyear) control_cohort(lastcohort)
    plot_event_study_SA, graph_name("`var'") graph_title("`var'") subfolder("main")
    
    // With firm-year trends and industry-year FE
    eventstudyinteract `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        vce(cluster city_code) absorb(org_ID_num#c.year year#monitor_id year#bindustry) cohort(openyear) control_cohort(lastcohort)
    plot_event_study_SA_IYFE, graph_name("`var'") graph_title("`var'") subfolder("main")
}

// PPML estimation with firm-year trends
local ppml_vars "so2_emission nox_emission cod_emission nh3n_emission"
foreach var of local ppml_vars {
    // With firm-year trends
    ppmlhdfe `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        absorb(org_ID_num#c.year year#monitor_id) cluster(city_code)
    estimates store ols
    plot_event_study_ppml, graph_name("`var'") graph_title("`var'") subfolder("main")
    
    // With firm-year trends and industry-year FE
    ppmlhdfe `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        absorb(org_ID_num#c.year year#monitor_id year#bindustry) cluster(city_code)
    estimates store ols
    plot_event_study_ppml_IYFE, graph_name("`var'") graph_title("`var'") subfolder("main")
}

/*------------------------------------------------------------------------------
* SECTION 4: EXTENDED DATASET ANALYSIS (merged_extended.dta)
*------------------------------------------------------------------------------*/

use "$workdata/merged_extended.dta", clear
prepare_event_data

// Main emission analysis for extended dataset
local emission_vars "log_so2_emission log_nox_emission"
foreach var of local emission_vars {
    // Baseline specification
    eventstudyinteract `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        vce(cluster city_code) absorb(org_ID_num year#monitor_id) cohort(openyear) control_cohort(lastcohort)
    plot_event_study_SA, graph_name("`var'") graph_title("`var'") subfolder("extended_using_cems")
    
    // With industry-year FE
    eventstudyinteract `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        vce(cluster city_code) absorb(org_ID_num year#bindustry year#monitor_id) cohort(openyear) control_cohort(lastcohort)
    plot_event_study_SA_IYFE, graph_name("`var'") graph_title("`var'") subfolder("extended_using_cems")
}

// PPML estimation for extended dataset
local ppml_vars "so2_emission nox_emission"
foreach var of local ppml_vars {
    // Baseline specification
    ppmlhdfe `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        absorb(org_ID_num year#monitor_id) cluster(city_code)
    estimates store ols
    plot_event_study_ppml, graph_name("`var'") graph_title("`var'") subfolder("extended_using_cems")
    
    // With industry-year FE
    ppmlhdfe `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        absorb(org_ID_num year#bindustry year#monitor_id) cluster(city_code)
    estimates store ols
    plot_event_study_ppml_IYFE, graph_name("`var'") graph_title("`var'") subfolder("extended_using_cems")
}

/*------------------------------------------------------------------------------
* SECTION 5: BALANCED PANEL ANALYSIS
*------------------------------------------------------------------------------*/

// Create balanced panel ID list
use year org_ID using "$data\Tax\tax_use_2008-2016_full.dta", clear
bysort org_ID: egen max_year = max(year)
keep if max_year == 2016
duplicates drop org_ID, force
save "$workdata/Workdata for emission data cleaning/balanced_org_ID.dta", replace

// Load merged data and restrict to balanced panel
use "$workdata/merged.dta", clear
bysort org_ID: egen max = max(year)
merge m:1 org_ID using "$workdata/Workdata for emission data cleaning/balanced_org_ID.dta"
keep if _merge == 3
drop _merge 

prepare_event_data


// Balanced panel analysis
local emission_vars "log_so2_emission log_nox_emission log_cod_emission log_nh3n_emission"
foreach var of local emission_vars {
    // With firm-year trends
    eventstudyinteract `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        vce(cluster city_code) absorb(org_ID_num#c.year year#monitor_id) cohort(openyear) control_cohort(lastcohort)
    plot_event_study_SA, graph_name("`var'") graph_title("`var'") subfolder("balanced")
    
    // With firm-year trends and industry-year FE
    eventstudyinteract `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        vce(cluster city_code) absorb(org_ID_num#c.year year#monitor_id year#bindustry) cohort(openyear) control_cohort(lastcohort)
    plot_event_study_SA_IYFE, graph_name("`var'") graph_title("`var'") subfolder("balanced")
}

// PPML for balanced panel
local ppml_vars "so2_emission nox_emission industrial_output cod_emission nh3n_emission"
foreach var of local ppml_vars {
    // With firm-year trends
    ppmlhdfe `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        absorb(org_ID_num#c.year year#monitor_id) cluster(city_code)
    estimates store ols
    plot_event_study_ppml, graph_name("`var'") graph_title("`var'") subfolder("balanced")
    
    // With firm-year trends and industry-year FE
    ppmlhdfe `var' dis_1_* dis_2_* dis_3_* dis_4_* if distance_km <= 50, ///
        absorb(org_ID_num#c.year year#monitor_id year#bindustry) cluster(city_code)
    estimates store ols
    plot_event_study_ppml_IYFE, graph_name("`var'") graph_title("`var'") subfolder("balanced")
}

/*==============================================================================
* END OF FILE 1
*==============================================================================*/
