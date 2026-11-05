using System.Globalization;

public static class CsvLoader
{
    // Reads the cleaned CSV into a list of Product objects. Time: O(n)
    public static List<Product> Load(string path)
    {
        var list = new List<Product>();
        foreach (var line in File.ReadLines(path).Skip(1)) // skip header row
        {
            var c = line.Split(',');
            list.Add(new Product {
                Id = int.Parse(c[0]), Name = c[1],
                Price = double.Parse(c[2], CultureInfo.InvariantCulture),
                Score = double.Parse(c[3], CultureInfo.InvariantCulture),
                Rating = double.Parse(c[4], CultureInfo.InvariantCulture),
                Niche = c[5] });
        }
        return list;
    }
}