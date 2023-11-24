to_process = 'to_process'
need_show_sifted_figures = 'need_show_sifted_figures'
print_forecast_out = 'print_forecast_out'
shift_percent = 'shift_percent'
save_to_excel = 'save_to_excel'
directory = 'dir'
test_size = 'test_size'
random_state = 'random_state'
show_forecast = 'show_forecast'
save_png = 'save_png'

show_all_history = False

settings = {
    'VTB': {
        to_process: True,
        save_to_excel: True,
        directory: 'втб/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 42
    },
    'Доллар': {
        to_process: False,
        save_to_excel: True,
        directory: 'доллар/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 50
    },
    'Brent': {
        to_process: False,
        save_to_excel: True,
        directory: 'brent/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 43
    },
    'Gazprom': {
        to_process: False,
        save_to_excel: True,
        directory: 'газпром/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 41
    },
    'Лукойл': {
        to_process: False,
        save_to_excel: True,
        directory: 'лукойл/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: False,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 40
    },
    'Роснефть': {
        to_process: False,
        save_to_excel: True,
        directory: 'роснефть/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 41
    },
    'Сбербанк': {
        to_process: False,
        save_to_excel: True,
        directory: 'сбербанк/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 42
    },
    'Сургутнефтегаз': {
        to_process: False,
        save_to_excel: True,
        directory: 'сургутнефтегаз/',
        need_show_sifted_figures: False,
        print_forecast_out: True,
        show_forecast: True,
        save_png: True,
        shift_percent: .01,
        test_size: .2,
        random_state: 44
    }
}
