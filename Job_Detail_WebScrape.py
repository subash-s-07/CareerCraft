import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd

list_url = "https://www.linkedin.com/jobs/search?keywords=Data%20Analysis&location=India"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

async def fetch(session, url):
    """Fetch the content of the URL asynchronously with headers."""
    async with session.get(url, headers=HEADERS) as response:
        return await response.text()

async def extract_job_ids():
    """Extract job IDs from the search page."""
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, list_url)
        soup = BeautifulSoup(html, "html.parser")
        job_list = soup.find('ul', class_='jobs-search__results-list')

        if not job_list:
            return []

        job_ids = []
        for job in job_list.find_all('li'):
            job_div = job.find('div', attrs={"data-entity-urn": True})
            if job_div:
                job_id = job_div["data-entity-urn"].split(":")[-1]
                job_ids.append(job_id)

        return job_ids  # Limit to 5 for testing

async def fetch_job_descriptions(job_ids):
    """Fetch job descriptions asynchronously."""
    job_hash = {
        'Job_ID': [], 'Role': [], 'Seniority': [], 'Emp_Type': [],
        'Job function': [], 'Industries': [], 'Job_Des': [], 'Link': []
    }

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, f'https://www.linkedin.com/jobs/view/{job_id}/') for job_id in job_ids]
        responses = await asyncio.gather(*tasks)

        for job_id, html in zip(job_ids, responses):
            soup = BeautifulSoup(html, "html.parser")

            try:
                job_desc = soup.find('div', class_='show-more-less-html__markup')
                role = soup.find('h1', class_='top-card-layout__title')
                criteria = soup.find_all('span', class_='description__job-criteria-text')

                job_hash['Job_ID'].append(job_id)
                job_hash['Role'].append(role.text.strip())
                job_hash['Seniority'].append(criteria[0].text.strip())
                job_hash['Emp_Type'].append(criteria[1].text.strip())
                job_hash['Job function'].append(criteria[2].text.strip())
                job_hash['Industries'].append(criteria[3].text.strip())
                job_hash['Job_Des'].append(job_desc.text.strip())
                job_hash['Link'].append(f'https://www.linkedin.com/jobs/view/{job_id}/')

            except Exception as e:
                print(f"Error processing job {job_id}: {e}")


    return job_hash

async def main():
    job_ids = await extract_job_ids()
    job_data = await fetch_job_descriptions(job_ids)
    pd.DataFrame(job_data).to_csv('linkedin_jobs.csv', index=False)

# Run the async functions
asyncio.run(main())
