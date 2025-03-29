from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import pandas as pd

def scrape_linkedin_jobs(job_list):
    # Set up WebDriver (Ensure geckodriver path is correct)
    """service = Service("path/to/geckodriver")  
    driver = webdriver.Firefox(service=service)"""
    driver = webdriver.Firefox()

    # Open Indeed Salary Page
    driver.get("https://www.indeed.com/career/salaries/")

    job_descriptions = []

    for job_title in job_list:
        try:
            # Wait for and enter job title
            title_box = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "input-title-autocomplete"))
            )
            title_box.clear()
            title_box.send_keys(job_title)

            # Wait for and enter location
            location_box = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "input-location-autocomplete"))
            )
            location_box.clear()
            location_box.send_keys("United States")

            # Click Search Button
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "title-location-search-btn"))
            )
            search_button.send_keys(Keys.RETURN)

            # Wait for the page to load
            time.sleep(5)

            # Click clear buttons
            try:
                clear_title = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "clear-title-localized"))
                )
                driver.execute_script("arguments[0].click();", clear_title)
            except:
                pass  # Ignore if not found

            try:
                clear_location = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "clear-location-localized"))
                )
                driver.execute_script("arguments[0].click();", clear_location)
            except:
                pass  # Ignore if not found

            # Select "Per year" in the dropdown
            dropdown_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "pay-period-selector"))
            )
            dropdown = Select(dropdown_element)
            dropdown.select_by_visible_text("Per year")

            time.sleep(5)

            # Extract salary
            salary_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-testid='avg-salary-value']"))
            )
            salary_text = salary_element.text
            salary = f"The Salary of {job_title} is {salary_text}"
            job_descriptions.append(salary)

        except Exception as e:
            print(f"Error during job scraping for {job_title}: {e}")

    driver.quit()
    return job_descriptions


if __name__ == "__main__":
    job_list = ["Machine Learning Engineer", "Data Scientist", "Computer Vision",
                "Data Analyst", "Python Developer", "Data Engineer"]

    descriptions = scrape_linkedin_jobs(job_list)
    df = pd.DataFrame(descriptions, columns=["Job Salaries"])
    
    # Save to CSV
    csv_file = "job_listings.csv"
    try:
        df.to_csv(csv_file, index=False, mode="a", header=False, encoding="utf-8")
        print(f"Job data appended to '{csv_file}'.")
    except Exception as e:
        print(f"Error appending to CSV: {e}")
