CC := g++ # This is the main compiler
SRCDIR := src
BUILDDIR := build
TEST_SRCDIR := test
TEST_BUILDDIR := build/test
TARGET := bin/scnn
TEST := bin/test
TMP := test/data/tmp

SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.$(SRCEXT)=.o))
TEST_SOURCES := $(shell find $(TEST_SRCDIR) -type f -name *.$(SRCEXT))
TEST_OBJECTS := $(patsubst $(TEST_SRCDIR)/%, $(TEST_BUILDDIR)/%, $(TEST_SOURCES:.$(SRCEXT)=.o))
CFLAGS := --std=c++11 -g # -Wall
LIB := -L lib
INC := -I include
RM := rm

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

# Tests
test: $(TEST_OBJECTS) $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TEST) $(LIB)"; $(CC) $^ -o $(TEST) $(LIB)

$(TEST_BUILDDIR)/%.o: $(TEST_SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(TEST_BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

run:
	@$(TARGET)

runtest:
	@$(TEST)

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET) $(TEST)"; $(RM) -r $(BUILDDIR) $(TARGET) $(TEST)