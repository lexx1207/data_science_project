import numpy as np
def game_core_v3(number: int = 1) -> int:
    """
    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
   
    count = 0
    predict = np.random.randint(1, 101)
    range_number = [1,101] # Создаем список из 2 элементов которые будут являться границами при генерации числа
    
    while number != predict:
        count += 1
        
        if number > predict:
          range_number[0] = predict # Задаем нижнюю границу для следующей генерации числа 
          predict = np.random.randint(range_number[0], range_number[1])
        else:
          range_number[1] = predict # Задаем верхнюю границу для следующей генерации числа
          predict = np.random.randint(range_number[0], range_number[1])
    
    return count

def score_game(random_predict) -> int:
    """За какое количество попыток в среднем за 10000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    #np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(10000))  # загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за: {score} попытки")
    
if __name__ == '__main__':
    score_game(game_core_v3)