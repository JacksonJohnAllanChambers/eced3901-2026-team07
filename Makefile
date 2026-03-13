MCU = atmega328p
CC = avr-gcc
CFLAGS = -mmcu=$(MCU) -Wall -Os
PORT = /dev/ttyUSB0
BAUD = 115200

SRC = src/main.c src/lighting.c
TARGET = output

all: $(TARGET).hex

$(TARGET).elf: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

$(TARGET).hex: $(TARGET).elf
	avr-objcopy -O ihex $< $@

flash: $(TARGET).hex
	avrdude -c arduino -p $(MCU) -P $(PORT) -b $(BAUD) -U flash:w:$<

clean:
	rm -f $(TARGET).elf $(TARGET).hex