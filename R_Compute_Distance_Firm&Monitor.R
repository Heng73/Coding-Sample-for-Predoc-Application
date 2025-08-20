# ==================================================
# Compute distance pairings between firms (source: taxsurvey) and the monitor station
#   - Keep all pairings within 20 km
#   - If every station for a firm is > 20 km away, keep only the nearest one
# ==================================================
# Source: 
#     Firm -- taxsurvey
#     monitor -- 1. full list
#     monitor -- 2. china_monitor.csv list


# Part 1: Compute using monitor (full list)
library(sf)
library(dplyr)
library(readr)
library(purrr)
library(stringr)
library(haven)

setwd("/Users/tangheng/Dropbox/Green TFP China")

firm_path        <- file.path("Data/Workdata/Workdata for wind process", "firm_coord_taxsurvey_pollution_industry.csv")
station_list_path<- file.path("RA_Heng_work/Data/Monitor", "monitor_List_Opendate_Wind_data.csv")
out_dir          <- "RA_Heng_work/Data/Monitor"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 1. read data
## 1.1 firm (remove missing coordinates)
firms <- read_csv(firm_path,
                  col_types = cols(
                    org_ID       = col_character(),
                    indus       = col_character(),
                    bindustry    = col_character(),
                    COMPANY_ID = col_character(),
                    longitude    = col_double(),
                    latitude     = col_double(),
                    province_name  = col_character(),
                    city_name  = col_character(),
                    city_code  = col_character()
                  )) %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  mutate(
    firm_longitude = longitude,
    firm_latitude  = latitude
  )


## 1.2 Station
stations <- read_csv(station_list_path,
                     col_types = cols(
                       monitor_ID   = col_character(),
                       Monitor_Name = col_character(),
                       City         = col_character(),
                       longitude    = col_double(),
                       latitude     = col_double(),
                       openyear         = col_integer(),
                       openmonth        = col_integer()
                     )) %>%
  mutate(
    monitor_longitude = longitude,
    monitor_latitude  = latitude
  )

# 2. Build sf objects and project to meters (Web Mercator）
firms_sf <- st_as_sf(firms,
                     coords = c("longitude", "latitude"),
                     crs    = 4326) %>%
  st_transform(3857)

stations_sf <- st_as_sf(stations,
                        coords = c("longitude", "latitude"),
                        crs    = 4326) %>%
  st_transform(3857)

# 3.1 Spatial index — find pairs within 20 km
within20km <- st_is_within_distance(
  x    = firms_sf,
  y    = stations_sf,
  dist = 20000   # meter
)

# 3.2 Generate all (org_ID, monitor_ID, distance) pairs within 20 km
pairs_in <- imap_dfr(within20km, function(idxs, i) {
  if (length(idxs) == 0) return(NULL)
  dists <- st_distance(
    firms_sf[i, ],
    stations_sf[idxs, ],
    by_element = FALSE
  )
  tibble(
    org_ID     = firms$org_ID[i],
    monitor_ID = stations$monitor_ID[idxs],
    distance   = as.numeric(dists) / 1000  # meter -> km
  )
})

# 4. For firms with no stations within 20 km, keep only the nearest station
no_match_idx <- which(lengths(within20km) == 0)
if (length(no_match_idx) > 0) {
  all_dists <- st_distance(
    firms_sf[no_match_idx, ],
    stations_sf,
    by_element = FALSE
  )
  min_idx  <- apply(all_dists, 1, which.min)
  min_dist <- apply(all_dists, 1, min)
  pairs_out <- tibble(
    org_ID     = firms$org_ID[no_match_idx],
    monitor_ID = stations$monitor_ID[min_idx],
    distance   = as.numeric(min_dist) / 1000
  )
} else {
  pairs_out <- tibble(org_ID = character(),
                      monitor_ID = character(),
                      distance = double())
}

# 5. Combine all pairs
all_pairs <- bind_rows(pairs_in, pairs_out) %>%
  rename(distance_monitor_firm = distance)

# 6. Add firm and station attributes
firms_basic <- firms %>%
  select(org_ID, COMPANY_ID, province_name, city_name, city_code,
         firm_longitude, firm_latitude)

stations_basic <- stations %>%
  select(monitor_ID, Monitor_Name, City, distance_monitor_station,
         WND_direction, openyear, openmonth, STATION,
         monitor_longitude, monitor_latitude, days_count, total_days)

result <- all_pairs %>%
  left_join(firms_basic,    by = "org_ID") %>%
  left_join(stations_basic, by = "monitor_ID") %>%
  select(
    org_ID,
    COMPANY_ID,
    firm_longitude,
    firm_latitude,
    monitor_ID,
    Monitor_Name,
    monitor_longitude,
    monitor_latitude,
    City,
    openyear,
    openmonth,
    distance_monitor_firm,
    distance_monitor_station,
    STATION,
    WND_direction,
    days_count,      
    total_days,
    province_name,
    city_name,
    city_code
  )

# 7. Creat `openquarter` for quarter-panel event study
result_unique <- result %>%
  filter(!is.na(distance_monitor_firm)) %>%
  # (1) Create a single numeric value representing "station opening date"
  mutate(open_date_num = openyear * 100 + openmonth) %>%
  group_by(org_ID) %>%
  # (2) Keep the earliest opening date
  filter(open_date_num == min(open_date_num, na.rm = TRUE)) %>%
  # (3) Then keep the smallest distance
  filter(distance_monitor_firm == min(distance_monitor_firm, na.rm = TRUE)) %>%
  # (4) If still multiple rows, take the first one
  slice_head(n = 1) %>%
  ungroup() %>%
  # Remove helper column and create `openquarter`
  mutate(
    openquarter = ( (openmonth - 1) %/% 3 ) + 1
  ) %>%
  select(-open_date_num)

# 8. Output final CSV
out_file_dta <- file.path(out_dir, "earliest_closest_taxsurvey_chinamonitor(withOpenmonth).dta")
write_dta(result_unique, out_file_dta)


# ---------------------------------------------------------------------------------------------


# Part 2: Compute using monitor (china_monitor.csv list)
library(sf)
library(dplyr)
library(readr)
library(purrr)
library(stringr)

setwd("/Users/tangheng/Dropbox/Green TFP China")

firm_path        <- file.path("Data/Workdata/Workdata for wind process", "firm_coord_taxsurvey_pollution_industry.csv")
station_list_path<- file.path("RA_Heng_work/Data/Monitor", "monitor_List(china)_Opendate_Wind_data.csv")
out_dir          <- "RA_Heng_work/Data/Monitor"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# 1. read data
## 1.1 firm (remove missing coordinates)
firms <- read_csv(firm_path,
                  col_types = cols(
                    org_ID       = col_character(),
                    indus       = col_character(),
                    bindustry    = col_character(),
                    COMPANY_ID = col_character(),
                    longitude    = col_double(),
                    latitude     = col_double(),
                    province_name  = col_character(),
                    city_name  = col_character(),
                    city_code  = col_character()
                  )) %>%
  filter(!is.na(longitude) & !is.na(latitude)) %>%
  mutate(
    firm_longitude = longitude,
    firm_latitude  = latitude
  )

## 1.2 Station
stations <- read_csv(station_list_path,
                     col_types = cols(
                       monitor_ID   = col_character(),
                       Monitor_Name = col_character(),
                       City         = col_character(),
                       longitude    = col_double(),
                       latitude     = col_double(),
                       openyear         = col_integer()
                     )) %>%
  mutate(
    monitor_longitude = longitude,
    monitor_latitude  = latitude
  )

# 2. Build sf objects and project to meters (Web Mercator）
firms_sf <- st_as_sf(firms,
                     coords = c("longitude", "latitude"),
                     crs    = 4326) %>%
  st_transform(3857)

stations_sf <- st_as_sf(stations,
                        coords = c("longitude", "latitude"),
                        crs    = 4326) %>%
  st_transform(3857)

# 3.1 Spatial index — find pairs within 20 km
within20km <- st_is_within_distance(
  x    = firms_sf,
  y    = stations_sf,
  dist = 20000   # meter
)

# 3.2 Generate all (org_ID, monitor_ID, distance) pairs within 20 km
pairs_in <- imap_dfr(within20km, function(idxs, i) {
  if (length(idxs) == 0) return(NULL)
  dists <- st_distance(
    firms_sf[i, ],
    stations_sf[idxs, ],
    by_element = FALSE
  )
  tibble(
    org_ID     = firms$org_ID[i],
    monitor_ID = stations$monitor_ID[idxs],
    distance   = as.numeric(dists) / 1000  # meter -> km
  )
})

# 4. For firms with no stations within 20 km, keep only the nearest station
no_match_idx <- which(lengths(within20km) == 0)
if (length(no_match_idx) > 0) {
  all_dists <- st_distance(
    firms_sf[no_match_idx, ],
    stations_sf,
    by_element = FALSE
  )
  min_idx  <- apply(all_dists, 1, which.min)
  min_dist <- apply(all_dists, 1, min)
  pairs_out <- tibble(
    org_ID     = firms$org_ID[no_match_idx],
    monitor_ID = stations$monitor_ID[min_idx],
    distance   = as.numeric(min_dist) / 1000
  )
} else {
  pairs_out <- tibble(org_ID = character(),
                      monitor_ID = character(),
                      distance = double())
}

# 5. Combine all pairs
all_pairs <- bind_rows(pairs_in, pairs_out) %>%
  rename(distance_monitor_firm = distance)

# 6. Add firm and station attributes
firms_basic <- firms %>%
  select(org_ID, COMPANY_ID, province_name, city_name, city_code,
         firm_longitude, firm_latitude)

stations_basic <- stations %>%
  select(monitor_ID, stationid_s, Monitor_Name, City,
         distance_monitor_station, WND_direction, openyear, STATION,
         monitor_longitude, monitor_latitude, days_count, total_days)

result <- all_pairs %>%
  left_join(firms_basic,    by = "org_ID") %>%
  left_join(stations_basic, by = "monitor_ID") %>%
  select(
    org_ID,
    COMPANY_ID,
    firm_longitude,
    firm_latitude,
    monitor_ID,
    stationid_s,
    Monitor_Name,
    City,
    monitor_longitude,
    monitor_latitude,
    openyear,
    distance_monitor_firm,
    distance_monitor_station,
    STATION,
    WND_direction,
    days_count,      
    total_days,
    province_name,
    city_name,
    city_code
  ) %>%
  mutate(
    COMPANY_ID = str_remove(COMPANY_ID, "\\.0+$")
  )

# 7. Output final CSV
out_file <- file.path(out_dir, "Distance_firm(taxsurvey)_monitor(china)_pairs.csv")
write_csv(result, out_file)

message("Finished：", out_file)



