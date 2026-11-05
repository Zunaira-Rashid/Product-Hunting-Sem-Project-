var path = Path.Combine(AppContext.BaseDirectory, "../../../sample_products.csv");
var products = CsvLoader.Load(path);
Console.WriteLine($"Loaded {products.Count} products");
foreach (var p in products.Take(5))
    Console.WriteLine($"{p.Id} | {p.Name} | ${p.Price} | score {p.Score}");