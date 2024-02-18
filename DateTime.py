#To find one day date
import datetime
print("the Date Today is:",datetime.datetime.today())
date_today=datetime.date.today()
print(date_today)

#ist to find the date before four days,after four days
import datetime
day1=datetime.date(2018,2,12)
print('day1:',day1.ctime())
day2=datetime.date(2018,2,12)
print('day2:',day2.ctime())
print('number of days:',day1-day2)
date_today=datetime.date.today()
no_of_days=datetime.timedelta(days=4)
before_four_days=date_today-no_of_days
print('before four days:',before_four_days)
after_four_days=date_today+no_of_days
print('after four days:',after_four_days)

