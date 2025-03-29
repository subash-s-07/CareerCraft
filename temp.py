import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd

# Base URL for job listings
LIST_URL = 'https://www.linkedin.com/jobs/search?keywords=Machine%20Learning&location=India&geoId=102713980&position=1&pageNum=0'

# Headers to mimic a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
}

async def fetch(session, url):
    """Fetch a URL asynchronously"""
    async with session.get(url, headers=HEADERS) as response:
        return await response.text()

async def get_job_ids():
    """Fetch job IDs from the LinkedIn job search page."""
    async with aiohttp.ClientSession() as session:
        page_data = await fetch(session, LIST_URL)
        soup = BeautifulSoup(page_data, "html.parser")
        job_list = soup.find('ul', class_='jobs-search__results-list')

        if not job_list:
            return []
        
        jobs = job_list.find_all('li')
        job_ids = []

        for job in jobs:
            job_div = job.find('div', attrs={"data-entity-urn": True})
            if job_div:
                job_id = job_div["data-entity-urn"].split(":")[-1]
                job_ids.append(job_id)
        
        return job_ids  # Limit to first 5 jobs for efficiency

async def fetch_job_details(job_id, session):
    """Fetch job details for a given job ID."""
    job_url = f'https://www.linkedin.com/jobs/view/{job_id}/'
    page_data = await fetch(session, job_url)
    soup = BeautifulSoup(page_data, "html.parser")

    try:
        job_des = soup.find('div', class_='show-more-less-html__markup').text.strip()
        role = soup.find('h1', class_='top-card-layout__title').text.strip()
        criteria = soup.find_all('span', class_='description__job-criteria-text')
        
        return {
            'Job_ID': job_id,
            'Role': role,
            'Seniority': criteria[0].text.strip() if len(criteria) > 0 else "N/A",
            'Emp_Type': criteria[1].text.strip() if len(criteria) > 1 else "N/A",
            'Job function': criteria[2].text.strip() if len(criteria) > 2 else "N/A",
            'Industries': criteria[3].text.strip() if len(criteria) > 3 else "N/A",
            'Job_Des': job_des,
            'Link': job_url
        }
    except:
        return None  # Skip if parsing fails

async def main():
    job_ids = await get_job_ids()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_job_details(job_id, session) for job_id in job_ids]
        job_data = await asyncio.gather(*tasks)
    
    job_data = [job for job in job_data if job]  # Remove None values
    df = pd.DataFrame(job_data)
    df.to_csv('jobs.csv')

# Run the async loop
asyncio.run(main())
