import requests


url = 'https://stats.espncricinfo.com/ci/engine/records/team/results_summary.html?id=89;type=trophy'
r = requests.get(url)
content = r.content
print(content)

