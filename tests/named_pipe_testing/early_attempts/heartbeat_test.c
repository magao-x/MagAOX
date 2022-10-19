#include <time.h>
#include <stdio.h>
int main() {
time_t future;
time_t now;
unsigned long ct = 0;
  printf("%d ... ", sizeof(time_t));
  future = 10 + time(NULL);
  while ((now=time(NULL))<=future) ++ct;
  printf("%lu %lu %lu \n", future, now, ct);
  return 0;
}
