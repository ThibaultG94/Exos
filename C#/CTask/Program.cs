using System;
using System.Collections.Generic;

class Program
{
    static List<string> tasks = new List<string>();

    static void Main()
    {
        while (true)
        {
            Console.Clear();
            Console.WriteLine("=== Gestionnaire de Tâches ===");
            Console.WriteLine("1. Ajouter une tâche");
            Console.WriteLine("2. Afficher les tâches");
            Console.WriteLine("3. Supprimer une tâche");
            Console.WriteLine("4. Quitter");
            Console.Write("Choix : ");
            
            string choix = Console.ReadLine();

            switch (choix)
            {
                case "1":
                    AjouterTache();
                    break;
                case "2":
                    AfficherTaches();
                    break;
                case "3":
                    SupprimerTache();
                    break;
                case "4":
                    Environment.Exit(0);
                    break;
                default:
                    Console.WriteLine("Choix invalide !");
                    break;
            }
        }
    }

    static void AjouterTache()
    {
        Console.Write("Nouvelle tâche : ");
        string tache = Console.ReadLine();
        tasks.Add(tache);
        Console.WriteLine("Tâche ajoutée !");
        Console.ReadKey();
    }

    static void AfficherTaches()
    {
        Console.WriteLine("=== Liste des Tâches ===");
        if (tasks.Count == 0)
        {
            Console.WriteLine("Aucune tâche enregistrée.");
        }
        else
        {
            for (int i = 0; i < tasks.Count; i++)
            {
                Console.WriteLine($"{i + 1}. {tasks[i]}");
            }
        }
        Console.ReadKey();
    }

    static void SupprimerTache()
    {
        AfficherTaches();
        Console.Write("Numéro de la tâche à supprimer : ");
        if (int.TryParse(Console.ReadLine(), out int index) && index > 0 && index <= tasks.Count)
        {
            tasks.RemoveAt(index - 1);
            Console.WriteLine("Tâche supprimée !");
        }
        else
        {
            Console.WriteLine("Numéro invalide !");
        }
        Console.ReadKey();
    }
}
