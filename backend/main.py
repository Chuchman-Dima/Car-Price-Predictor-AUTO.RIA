import requests
import pandas as pd
import os
import time

# Твій ключ (порада: краще зберігати його в змінних середовища для безпеки)
API_KEY = '0JddtV85CWYfUE56u9oO8q0mGvTl7jlrSp4QNsi3'
FILE_NAME = '../data/cars_dataset.csv'
MAX_REQUESTS = 2000


def get_collected_ids():
    """Повертає множину вже завантажених ID з файлу."""
    if os.path.exists(FILE_NAME):
        return set(pd.read_csv(FILE_NAME)['ID'].astype(str))
    return set()


collected = get_collected_ids()
print(f"Вже зібрано: {len(collected)} авто.")

requests_made = 0
to_download = []
page = 0

print("Шукаємо нові авто...")

# ЕТАП 1: Збираємо ID, гортаючи сторінки пошуку
while requests_made < MAX_REQUESTS:
    search_url = f'https://developers.ria.com/auto/search?api_key={API_KEY}&category_id=1&countpage=100&page={page}'

    try:
        search_res = requests.get(search_url)
        requests_made += 1  # Рахуємо пошуковий запит
    except Exception as e:
        print(f"Помилка під час пошуку: {e}")
        break

    if search_res.status_code == 200:
        result_data = search_res.json().get('result', {}).get('search_result', {})
        all_ids = result_data.get('ids', [])

        if not all_ids:
            print("Більше немає сторінок для пошуку (досягли кінця результатів).")
            break

        # Відбираємо тільки нові ID, яких ще немає у файлі
        new_ids = [str(i) for i in all_ids if str(i) not in collected]

        # Додаємо нові унікальні ID до загальної черги завантаження
        for car_id in new_ids:
            if car_id not in to_download:
                to_download.append(car_id)

        # Скільки ще запитів ми можемо зробити для отримання деталей авто?
        remaining_requests = MAX_REQUESTS - requests_made

        # Якщо ми зібрали достатньо ID, обрізаємо список до потрібної кількості і виходимо
        if len(to_download) >= remaining_requests:
            to_download = to_download[:remaining_requests]
            break

        page += 1
        time.sleep(0.5)  # Маленька пауза між перегортанням сторінок
    else:
        print(f"Пошук не вдався. Код: {search_res.status_code}")
        break

# ЕТАП 2: Завантажуємо деталі для знайдених ID
if not to_download:
    print("Нових авто не знайдено або всі доступні вже в базі.")
else:
    print(f"Знайдено {len(to_download)} нових автомобілів. Починаємо завантаження деталей...")

    for car_id in to_download:
        if requests_made >= MAX_REQUESTS:
            break

        info_url = f'https://developers.ria.com/auto/info?api_key={API_KEY}&auto_id={car_id}'

        try:
            res = requests.get(info_url)
            requests_made += 1  # Рахуємо запит на деталі
        except Exception as e:
            print(f"Помилка запиту для ID {car_id}: {e}")
            time.sleep(3)
            continue

        if res.status_code == 200:
            data = res.json()

            # Обробка даних із захистом від відсутності полів
            mark_name = data.get('markName', '')
            model_name = data.get('modelName', '')
            auto_data = data.get('autoData', {})

            car_entry = {
                'ID': car_id,
                'Mark': mark_name,
                'Model': model_name,
                'Year': auto_data.get('year', 0),
                'Price_USD': data.get('USD', 0),
                'Mileage': auto_data.get('raceInt', 0),
                'Engine': auto_data.get('engineVolume', 0),
                'Fuel': auto_data.get('fuelName', ''),
                'Gearbox': auto_data.get('gearboxName', '')
            }

            df = pd.DataFrame([car_entry])
            df.to_csv(FILE_NAME, mode='a', index=False, header=not os.path.exists(FILE_NAME))
            print(f"[{requests_made}/{MAX_REQUESTS}] Збережено: {mark_name} {model_name} (ID: {car_id})")

            # Пауза 1 сек для дотримання стабільної швидкості
            time.sleep(1)
        else:
            print(f"Помилка на ID {car_id}: {res.status_code}")
            # Якщо API каже "Too Many Requests" (429), зупиняємось
            if res.status_code == 429:
                print("Перевищено ліміт запитів API (429 Too Many Requests).")
                break
            time.sleep(2)

    print("\n--- ГОТОВО ---")
    print(f"Використано запитів: {requests_made} із {MAX_REQUESTS}.")